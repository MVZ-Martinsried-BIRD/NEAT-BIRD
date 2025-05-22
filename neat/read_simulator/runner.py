"""
Runner for generate_reads task
"""

import time
import logging
import pickle
import gzip
from multiprocessing import Pool, cpu_count  # Added import

from Bio import SeqIO
from pathlib import Path

from .utils import (
    Options,
    parse_input_vcf,
    parse_beds,
    OutputFileWriter,
    generate_variants,
    generate_reads,
)
from ..common import validate_input_path, validate_output_path
from ..models import (
    MutationModel,
    SequencingErrorModel,
    FragmentLengthModel,
    TraditionalQualityModel,
)
from ..models.default_cancer_mutation_model import *
from ..variants import ContigVariants

from Bio.SeqRecord import SeqRecord  # Added import
from math import ceil  # Added import

__all__ = ["read_simulator_runner"]

_LOG = logging.getLogger(__name__)


# New helper function to generate reads for a chunk of a chromosome
def _generate_reads_for_chunk(args_tuple):
    """
    Generates reads for a given chunk of a chromosome.
    Designed to be called by multiprocessing.Pool.
    """
    (
        task_id,  # For logging, e.g., "chr1_chunk0"
        original_chrom_id,
        chunk_start_offset,
        reference_chunk_seqrecord,  # SeqRecord for the chunk
        variants_for_chromosome,  # ContigVariants for the entire original chromosome
        seq_error_model_1,
        seq_error_model_2,
        qual_score_model_1,
        qual_score_model_2,
        fraglen_model,
        target_regions_for_chromosome,  # Already filtered for the chromosome
        discard_regions_for_chromosome,  # Already filtered for the chromosome
        options,
    ) = args_tuple

    _LOG.debug(
        f"Generating reads for {task_id} ({original_chrom_id}:{chunk_start_offset}-{chunk_start_offset + len(reference_chunk_seqrecord.seq)})"
    )

    fastq_data, sam_order_data = generate_reads(
        reference=reference_chunk_seqrecord,
        error_model_1=seq_error_model_1,
        error_model_2=seq_error_model_2,
        qual_model_1=qual_score_model_1,
        qual_model_2=qual_score_model_2,
        fraglen_model=fraglen_model,
        contig_variants=variants_for_chromosome,  # Pass variants for the whole chromosome
        targeted_regions=target_regions_for_chromosome,  # Pass regions for the whole chromosome
        discarded_regions=discard_regions_for_chromosome,  # Pass regions for the whole chromosome
        options=options,
        chrom=original_chrom_id,  # Original chromosome ID
        ref_start=chunk_start_offset,  # Global start of this chunk
    )
    return fastq_data, sam_order_data


def initialize_all_models(options: Options):
    """
    Helper function that initializes models for use in the rest of the program.
    This includes loading the model and attaching the rng for this run
    to each model, so we can perform the various methods.

    :param options: the options for this run
    """

    # Load mutation model or instantiate default
    if options.mutation_model:
        mut_model = pickle.load(gzip.open(options.mutation_model))
    else:
        mut_model = MutationModel()

    # Set random number generator for the mutations:
    mut_model.rng = options.rng
    # Set custom mutation rate for the run, or set the option to the input rate so we can use it later
    if options.mutation_rate is not None:
        mut_model.avg_mut_rate = options.mutation_rate

    cancer_model = None
    if options.cancer and options.cancer_model:
        # cancer_model = pickle.load(gzip.open(options.cancer_model))
        # Set the rng for the cancer mutation model
        cancer_model.rng = options.rng
    elif options.cancer:
        # Note all parameters not entered here use the mutation madel defaults
        cancer_model = MutationModel(
            avg_mut_rate=default_cancer_avg_mut_rate,
            homozygous_freq=default_cancer_homozygous_freq,
            variant_probs=default_cancer_variant_probs,
            insert_len_model=default_cancer_insert_len_model,
            is_cancer=True,
        )

    _LOG.debug("Mutation models loaded")

    # We need sequencing errors to get the quality score attributes, even for the vcf
    if options.error_model:
        error_models = pickle.load(gzip.open(options.error_model))
        error_model_1 = error_models["error_model1"]
        quality_score_model_1 = error_models["qual_score_model1"]
        if options.paired_ended:
            if error_models["error_model2"]:
                error_model_2 = error_models["error_model2"]
                quality_score_model_2 = error_models["qual_score_model2"]
            else:
                _LOG.warning(
                    "Paired ended mode declared, but input sequencing error model is single ended,"
                    "duplicating model for both ends"
                )
                error_model_2 = error_models["error_model1"]
                quality_score_model_2 = error_models["qual_score_model1"]
        else:
            # ignore second model if we're in single-ended mode
            error_model_2 = None
            quality_score_model_2 = None
    else:
        # Use all the default values
        error_model_1 = SequencingErrorModel()
        quality_score_model_1 = TraditionalQualityModel()
        if options.paired_ended:
            error_model_2 = SequencingErrorModel()
            quality_score_model_2 = TraditionalQualityModel()
        else:
            error_model_2 = None
            quality_score_model_2 = None

    _LOG.debug("Sequencing error and quality score models loaded")

    if options.fragment_model:
        fraglen_model = pickle.load(gzip.open(options.fragment_model))
        fraglen_model.rng = options.rng
    elif options.fragment_mean:
        fraglen_model = FragmentLengthModel(
            options.fragment_mean, options.fragment_st_dev
        )
    else:
        # For single ended, fragment length will be based on read length
        fragment_mean = options.read_len * 2.0
        fragment_st_dev = fragment_mean * 0.2
        fraglen_model = FragmentLengthModel(fragment_mean, fragment_st_dev)

    _LOG.debug("Fragment length model loaded")

    return (
        mut_model,
        cancer_model,
        error_model_1,
        error_model_2,
        quality_score_model_1,
        quality_score_model_2,
        fraglen_model,
    )


def read_simulator_runner(config: str, output: str):
    """
    Run the generate_reads function, which generates simulated mutations in a dataset and corresponding files.
    Processes chromosomes sequentially, and chunks within chromosomes in parallel for read generation.
    FASTQ files are written per chromosome (appended). BAM and VCF are written globally at the end.

    :param config: This is a configuration file. Keys start with @ symbol. Everything else is ignored.
    :param output: This is the prefix for the output.
    """
    _LOG.debug(f"config = {config}")
    _LOG.debug(f"output = {output}")

    _LOG.info(f"Using configuration file {config}")
    config = Path(config).resolve()
    validate_input_path(config)

    # prepare output
    _LOG.info(f"Saving output files to {Path(output).parent}")
    output = Path(output).resolve()

    if not output.parent.is_dir():
        _LOG.info("Creating output dir")
        output.parent.mkdir(parents=True, exist_ok=True)

    # Read options file
    options = Options(output, config)

    # Validate output
    validate_output_path(output, False, options.overwrite_output)

    """
    Model preparation

    Read input models or default models, as specified by user.
    """
    _LOG.info("Reading Models...")

    (
        mut_model,
        cancer_model,
        seq_error_model_1,
        seq_error_model_2,
        qual_score_model_1,
        qual_score_model_2,
        fraglen_model,
    ) = initialize_all_models(options)

    """
    Process Inputs
    """
    _LOG.info(f"Reading {options.reference}.")

    # TODO check into SeqIO.index_db()
    reference_index = SeqIO.index(str(options.reference), "fasta")
    reference_keys_with_lens = {
        key: len(value) for key, value in reference_index.items()
    }
    _LOG.debug("Reference file indexed.")

    if _LOG.getEffectiveLevel() < 20:
        count = 0
        for contig in reference_keys_with_lens:
            count += reference_keys_with_lens[contig]
        _LOG.debug(f"Length of reference: {count / 1_000_000:.2f} Mb")

    input_variants_dict = {x: ContigVariants() for x in reference_keys_with_lens}
    if options.include_vcf:
        _LOG.info(f"Reading input VCF: {options.include_vcf}.")
        if options.cancer:
            # TODO Check if we need full ref index or just keys and lens
            sample_names = parse_input_vcf(
                input_variants_dict,
                options.include_vcf,
                options.ploidy,
                mut_model.homozygous_freq,
                reference_index,
                options,
                tumor_normal=True,
            )

            tumor_ind = sample_names["tumor_sample"]
            normal_ind = sample_names["normal_sample"]
        else:
            # TODO Check if we need full ref index or just keys and lens
            sample_names = parse_input_vcf(
                input_variants_dict,
                options.include_vcf,
                options.ploidy,
                mut_model.homozygous_freq,
                reference_index,
                options,
            )

        _LOG.debug("Finished reading input vcf file")

    # Note that parse_beds will return None for any empty or non-input files
    bed_files = (options.target_bed, options.discard_bed, options.mutation_bed)

    # parse input targeted regions, if present.
    if any(bed_files):
        _LOG.info(f"Reading input bed files.")

    (target_regions_dict, discard_regions_dict, mutation_rate_dict) = parse_beds(
        options, reference_keys_with_lens, mut_model.avg_mut_rate
    )

    if any(bed_files):
        _LOG.debug("Finished reading input beds.")

    # Prepare headers
    bam_header = None
    if options.produce_bam:
        # This is a dictionary that is the list of the contigs and the length of each.
        # This information will be needed later to create the bam header.
        bam_header = reference_keys_with_lens

    # Creates files and sets up objects for files that can be written to as needed.
    # Also creates headers for bam and vcf.
    # We'll also keep track here of what files we are producing.
    if options.cancer:
        output_normal = options.output.parent / f"{options.output.name}_normal"
        output_tumor = options.output.parent / f"{options.output.name}_tumor"
        output_file_writer = OutputFileWriter(options=options, bam_header=bam_header)
        output_file_writer_cancer = OutputFileWriter(
            options=options, bam_header=bam_header
        )
    else:
        outfile = options.output.parent / options.output.name
        output_file_writer = OutputFileWriter(options=options, bam_header=bam_header)
        output_file_writer_cancer = None

    _LOG.debug(f"Output files ready for writing.")

    """
    Begin Analysis
    """
    _LOG.info("Beginning simulation.")

    # Removed: breaks = find_file_breaks(reference_keys_with_lens)
    # _LOG.debug("Input reference partitioned for run") # This partitioning is now dynamic per chromosome

    global_variants_for_vcf = (
        {}
    )  # Store ContigVariants for each chromosome for the final VCF
    global_all_sam_order_data = []  # Aggregate all SAM order data for the final BAM

    # Determine number of processes for parallel chunk processing
    num_processes = (
        options.threads if options.threads and options.threads > 0 else cpu_count()
    )
    _LOG.info(
        f"Using {num_processes} processes for parallel read generation on chunks."
    )

    # Define a target chunk size (e.g., 10 Mbp, or make this configurable)
    # Adjusted to be an attribute of options or a constant
    target_chunk_size = getattr(options, "chunk_size", 10_000_000)

    # Iterate through chromosomes sequentially
    for original_chrom_id, original_chrom_len in reference_keys_with_lens.items():
        _LOG.info(
            f"Processing chromosome: {original_chrom_id} (length: {original_chrom_len})"
        )

        reference_chromosome_seq_record = reference_index[original_chrom_id]
        input_variants_for_chrom = input_variants_dict[original_chrom_id]
        mutation_rate_regions_for_chrom = mutation_rate_dict[original_chrom_id]
        target_regions_for_chrom = target_regions_dict[original_chrom_id]
        discard_regions_for_chrom = discard_regions_dict[original_chrom_id]

        # 1. Generate variants for the entire current chromosome
        _LOG.info(f"Generating variants for {original_chrom_id}")
        if options.paired_ended:
            max_qual_score = max(
                max(qual_score_model_1.quality_scores),
                max(qual_score_model_2.quality_scores) if qual_score_model_2 else 0,
            )
        else:
            max_qual_score = max(qual_score_model_1.quality_scores)

        variants_for_current_chrom = generate_variants(
            reference=reference_chromosome_seq_record,
            mutation_rate_regions=mutation_rate_regions_for_chrom,
            existing_variants=input_variants_for_chrom,
            mutation_model=mut_model,
            max_qual_score=max_qual_score,
            options=options,
        )
        global_variants_for_vcf[original_chrom_id] = variants_for_current_chrom
        _LOG.info(f"Finished generating variants for {original_chrom_id}")

        # 2. Define chunks for the current chromosome for parallel read generation
        chunks_for_this_chromosome = []
        num_chunks = max(1, ceil(original_chrom_len / target_chunk_size))
        actual_chunk_size = ceil(original_chrom_len / num_chunks)

        for i in range(num_chunks):
            chunk_start = i * actual_chunk_size
            chunk_end = min((i + 1) * actual_chunk_size, original_chrom_len)
            if chunk_start < chunk_end:  # Ensure chunk is not empty
                task_id = f"{original_chrom_id}_chunk{i}"
                # Create a SeqRecord for the chunk. ID should be original_chrom_id for Read object consistency.
                chunk_seq_slice = reference_chromosome_seq_record.seq[
                    chunk_start:chunk_end
                ]
                # Pass the original ID and description for the SeqRecord of the chunk
                chunk_seq_record = SeqRecord(
                    chunk_seq_slice,
                    id=original_chrom_id,
                    name=original_chrom_id,
                    description=reference_chromosome_seq_record.description,
                )
                chunks_for_this_chromosome.append(
                    (
                        task_id,
                        original_chrom_id,
                        chunk_start,  # This is ref_start for generate_reads
                        chunk_seq_record,
                        variants_for_current_chrom,  # Pass the variants for the whole chromosome
                        seq_error_model_1,
                        seq_error_model_2,
                        qual_score_model_1,
                        qual_score_model_2,
                        fraglen_model,
                        target_regions_for_chrom,  # Pass regions for the whole chromosome
                        discard_regions_for_chrom,  # Pass regions for the whole chromosome
                        options,
                    )
                )

        if not chunks_for_this_chromosome:
            _LOG.warning(
                f"No chunks generated for chromosome {original_chrom_id}, skipping read generation."
            )
            continue

        # 3. Generate reads in parallel for the chunks of the current chromosome
        _LOG.info(
            f"Generating reads for {len(chunks_for_this_chromosome)} chunks in {original_chrom_id} using {num_processes} processes..."
        )

        all_fastq_data_for_chrom = []
        # Use a context manager for the pool if possible, or ensure it's closed.
        # If num_processes is 1, Pool might not be necessary, can run serially.
        if num_processes > 1 and len(chunks_for_this_chromosome) > 1:
            with Pool(processes=num_processes) as pool:
                results_for_chrom_chunks = pool.map(
                    _generate_reads_for_chunk, chunks_for_this_chromosome
                )
        else:  # Run serially if only one process or one chunk
            results_for_chrom_chunks = [
                _generate_reads_for_chunk(args) for args in chunks_for_this_chromosome
            ]

        for fastq_data_chunk, sam_order_data_chunk in results_for_chrom_chunks:
            if fastq_data_chunk:
                all_fastq_data_for_chrom.append(fastq_data_chunk)
            if sam_order_data_chunk:
                global_all_sam_order_data.append(sam_order_data_chunk)

        _LOG.info(f"Finished generating reads for {original_chrom_id}.")

        # 4. Write FASTQ data for the current chromosome (appends for subsequent chromosomes)
        if options.produce_fastq and all_fastq_data_for_chrom:
            if options.paired_ended:
                _LOG.info(
                    f"Writing/Appending FASTQ data for {original_chrom_id} to: "
                    f"{', '.join([str(x) for x in output_file_writer.fastq_fns if x.name != 'dummy.fastq.gz']).strip(', ')}"
                )
            else:
                _LOG.info(
                    f"Writing/Appending FASTQ data for {original_chrom_id} to: {output_file_writer.fastq_fns[0]}"
                )
            output_file_writer.write_fastqs_from_memory(
                all_fastq_data_for_chrom, options.rng
            )
        elif options.produce_fastq:
            _LOG.info(
                f"No FASTQ data generated for chromosome {original_chrom_id} to write."
            )

    # After processing all chromosomes:
    # Write VCF (globally collected variants)
    if options.produce_vcf:
        _LOG.info(f"Outputting golden vcf: {str(output_file_writer.vcf_fn)}")
        output_file_writer.write_final_vcf(global_variants_for_vcf, reference_index)

    # Write BAM (globally collected SAM order data)
    if options.produce_bam:
        _LOG.info(f"Outputting golden bam file: {str(output_file_writer.bam_fn)}")
        contig_list = list(
            reference_keys_with_lens
        )  # Original contig list for BAM header
        contigs_by_index = {contig_list[n]: n for n in range(len(contig_list))}
        output_file_writer.output_bam_file(
            global_all_sam_order_data, contigs_by_index, options.read_len
        )

    _LOG.info("Read simulation finished.")


def find_file_breaks(reference_keys_with_lens: dict) -> dict:
    """
    Returns a dictionary with the chromosomes as keys, which is the start of building the chromosome map.
    This function is currently not used for parallelization in the new scheme but kept for potential other uses
    or if the old parallelization strategy is revisited.

    :param reference_keys_with_lens: a dictionary with chromosome keys and sequence values
    :return: a dictionary containing the chromosomes as keys and either "all" for values, or a list of indices
    """
    partitions = {}
    for contig in reference_keys_with_lens:
        partitions[contig] = [(0, reference_keys_with_lens[contig])]

    return partitions
