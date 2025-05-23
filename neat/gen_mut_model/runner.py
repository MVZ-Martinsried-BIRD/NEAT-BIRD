"""
Creates a mutation model
"""

import os.path
import pathlib
import pickle
import sys
import time  # Import the time module

import numpy as np
from Bio import SeqIO

from pathlib import Path
import logging

from ..models import MutationModel
from ..variants import *
from .constants import VCF_DEFAULT_POP_FREQ, DEF_HOMOZYGOUS_FRQ, DEF_MUT_SUB_MATRIX
from ..common import (
    validate_output_path,
    validate_input_path,
    open_input,
    open_output,
    ALLOWED_NUCL,
    ALL_TRINUCS,
    HUMAN_WHITELIST,
)
from .utils import (
    extract_header,
    read_and_filter_variants,
    cluster_list,
    count_trinucleotides,
    find_caf,
    convert_trinuc_transition_matrix,
    convert_snp_transition_matrix,
    convert_trinuc_mutation_dict,
    check_homozygous,
)

__all__ = ["compute_mut_runner"]

_LOG = logging.getLogger(__name__)


def runner(
    reference_index,
    vcf_to_process,
    vcf_columns,
    outcounts_file,
    show_trinuc,
    save_trinuc,
    output,
    bed,
    human_sample,
    skip_common,
    cores,  # Add cores parameter
):
    """
    This function generates the mutation model suitable for use in gen_reads. At the moment it must be run as a
    separate utility.

    :param dict reference_index: The reference index from SeqIO
    :param Path vcf_to_process: Path to the vcf to use for analysis
    :param list vcf_columns: Columns present in the input vcf
    :param Path outcounts_file: Path to outcounts file, if present
    :param bool show_trinuc: If true, display trinucleotide counts
    :param bool save_trinuc: If true, save trinucleotide counts
    :param Path output: Full path to final output
    :param Path bed: Optional input bed
    :param bool human_sample: If true, treat as human sample
    :param bool skip_common: if true, skip common
    :param int cores: Number of cores to use (added for consistency, though not used directly in timing)
    """
    start_time_total = time.time()

    # [(trinuc_a, trinuc_b)] = # of times we observed a mutation from trinuc_a into trinuc_b
    trinuc_transition_count = {}

    # total count of SNPs
    snp_count = 0
    # overall SNP transition probabilities
    snp_transition_count = {}
    # total count of insertions or deletions, indexed by length (negative is a deletion)
    indel_count = {}
    # Total homozygous variants detected
    homozygous_count = 0
    # detect variants that occur in a significant percentage of the input samples (pos,ref,alt,pop_fraction)
    common_variants = []
    # identify regions that have significantly higher local mutation rates than the average
    high_mut_regions = []

    # Clean up and simplify reference index
    # simplify naming and filter out actual human genomes from scaffolding
    ignore = []
    if human_sample:
        start_time_human_filter = time.time()
        for key in reference_index:
            if key not in HUMAN_WHITELIST:
                ignore.append(key)
        _LOG.info(
            f"Human sample filtering took {time.time() - start_time_human_filter:.4f} seconds"
        )

    if len(ignore) == len(reference_index):
        _LOG.error(
            "No valid human chromosome names detected. Check contig names reference."
        )
        sys.exit(1)

    # Pre-parsing to find all the matching chromosomes between ref and vcf
    _LOG.info("Processing VCF file...")
    start_time_read_filter = time.time()
    matching_variants, matching_chromosomes = read_and_filter_variants(
        vcf_to_process, reference_index, ignore
    )
    _LOG.info(
        f"read_and_filter_variants took {time.time() - start_time_read_filter:.4f} seconds"
    )

    if not matching_variants or not matching_chromosomes:
        _LOG.error(
            "No valid variants detected. Check names in vcf versus reference and/or bed."
        )
        sys.exit(1)

    start_time_count_trinuc = time.time()
    trinuc_ref_count, bed_track_length = count_trinucleotides(
        reference_index, bed, outcounts_file, matching_chromosomes, save_trinuc
    )
    _LOG.info(
        f"count_trinucleotides took {time.time() - start_time_count_trinuc:.4f} seconds"
    )

    if not trinuc_ref_count:
        _LOG.error("No valid trinucleotides detected in reference.")
        sys.exit(1)

    """
    Collect and analyze the data in the VCF file
    """
    _LOG.info("Creating mutational model...")
    total_reflen = 0
    # Remove any ignored contigs
    contigs_to_process = list(set(matching_chromosomes) - set(ignore))

    start_time_contig_loop = time.time()
    for contig in contigs_to_process:
        contig_start_time = time.time()
        # Running total of how many non-N bases there are in the reference
        total_reflen += len(reference_index[contig].seq) - reference_index[
            contig
        ].seq.upper().count("N")

        # list to be used for counting variants that occur multiple times in file (i.e. in multiple samples)
        vcf_common = {}

        # Create a list of variants to process
        variants_to_process = [x for x in matching_variants if x[0] == contig]

        if not variants_to_process:
            _LOG.warning(f"No variants found for {contig}")
            continue

        ref_sequence = reference_index[contig].seq.upper()

        # Count types
        variant_loop_start_time = time.time()
        find_caf_total_time = 0
        check_homozygous_total_time = 0
        for variant in variants_to_process:
            if variant[4] == "SNP":
                # Grab the nucleotides from the reference around this position
                trinuc_to_analyze = str(
                    ref_sequence[int(variant[1]) - 1 : int(variant[1]) + 2].upper()
                )
                # analyze is a trinuc from the reference
                if trinuc_to_analyze not in ALL_TRINUCS:
                    continue
                if variant[2] == trinuc_to_analyze[1]:
                    t_ref = trinuc_to_analyze
                    # t_alt changes the middle nuc to the ALT
                    t_alt = trinuc_to_analyze[0] + variant[3] + trinuc_to_analyze[2]
                    if t_alt not in ALL_TRINUCS:
                        continue

                    key = (t_ref, t_alt)
                    if key not in trinuc_transition_count:
                        trinuc_transition_count[key] = 0
                    trinuc_transition_count[key] += 1
                    snp_count += 1

                    # tracks transition probability of just the mutated nucleotide
                    key2 = (variant[2], variant[3])
                    if key2 not in snp_transition_count:
                        snp_transition_count[key2] = 0
                    snp_transition_count[key2] += 1

                else:
                    _LOG.error(
                        f"Ref allele in variant call does not match reference: {variant}"
                    )
                    sys.exit(1)

            else:
                indel_len = len(variant[3]) - len(variant[2])
                if indel_len not in indel_count:
                    indel_count[indel_len] = 0
                indel_count[indel_len] += 1

            find_caf_start = time.time()
            my_pop_freq = find_caf(variant[5])  # Use index 5 for INFO field
            find_caf_total_time += time.time() - find_caf_start

            check_homozygous_start = time.time()
            homozygous_count += check_homozygous(
                variant[5]
            )  # Use index 5 for INFO field
            check_homozygous_total_time += time.time() - check_homozygous_start

            vcf_common[tuple(variant[1:5])] = my_pop_freq
        _LOG.info(
            f"  Variant processing loop for {contig} took {time.time() - variant_loop_start_time:.4f} seconds"
        )
        _LOG.info(
            f"    Total time in find_caf for {contig}: {find_caf_total_time:.4f} seconds"
        )
        _LOG.info(
            f"    Total time in check_homozygous for {contig}: {check_homozygous_total_time:.4f} seconds"
        )
        # identify common mutations
        # common_mut_start_time = time.time()
        # percentile_var = 95
        # if vcf_common:  # Check if vcf_common is not empty
        #     min_value = np.percentile(
        #         [vcf_common[n] for n in vcf_common], percentile_var
        #     )
        #     for variant, allele_freq in sorted(vcf_common.items()):
        #         if allele_freq >= min_value:
        #             common_variants.append(variant)
        #             # TODO figure out what to do with these common variants
        # _LOG.info(
        #     f"  Common mutation identification for {contig} took {time.time() - common_mut_start_time:.4f} seconds"
        # )

        # Identify areas that have contained significantly higher random mutation rates.
        high_mut_start_time = time.time()
        # Added a potential for smaller deltas to handle smaller datasets in 4.0
        dist_thresh = min(2000, int(len(ref_sequence) * 0.01))
        percentile_clust = 97
        # Adjusted the scalar for smaller deltas in 4.0
        scaler = (
            min(1000, dist_thresh // 2) if dist_thresh > 0 else 1
        )  # Avoid division by zero
        # identify regions with disproportionately more variants in them
        if vcf_common:  # Check if vcf_common is not empty
            variant_pos = sorted([int(n[0]) for n in vcf_common])
            if variant_pos:  # Check if variant_pos is not empty
                cluster_list_start = time.time()
                clustered_pos = cluster_list(variant_pos, dist_thresh)
                _LOG.info(
                    f"    cluster_list took {time.time() - cluster_list_start:.4f} seconds"
                )
                # Since the list is sorted, taking the first and last position gives us the min and max
                by_len = [
                    (
                        len(clustered_pos[i]),
                        clustered_pos[i][0],
                        clustered_pos[i][-1],
                        i,
                    )
                    for i in range(len(clustered_pos))
                ]

                candidate_regions = []
                candidate_calc_start = time.time()
                for n in by_len:
                    ref_scalar = int((n[1] - dist_thresh) / float(scaler)) * scaler
                    alt_scalar = int((n[2] + dist_thresh) / float(scaler)) * scaler
                    # Avoid division by zero if ref_scalar == alt_scalar
                    denominator = float(alt_scalar - ref_scalar)
                    if denominator == 0:
                        density = 0.0  # Or handle as appropriate, e.g., np.inf or skip
                    else:
                        density = n[0] / denominator
                    candidate_regions.append(
                        (
                            density,
                            max([0, ref_scalar]),
                            min([len(ref_sequence), alt_scalar]),
                        )
                    )
                _LOG.info(
                    f"    Candidate region calculation loop took {time.time() - candidate_calc_start:.4f} seconds"
                )

                if candidate_regions:  # Check if candidate_regions is not empty
                    minimum_value = np.percentile(
                        [n[0] for n in candidate_regions], percentile_clust
                    )
                    high_mut_append_start = time.time()
                    for n in candidate_regions:
                        if n[0] >= minimum_value:
                            high_mut_regions.append((contig, n[1], n[2], n[0]))
                    _LOG.info(
                        f"    High mutation region appending loop took {time.time() - high_mut_append_start:.4f} seconds"
                    )

                    # collapse overlapping regions
                    collapse_start_time = time.time()
                    for i in range(len(high_mut_regions) - 1, 0, -1):
                        # Check bounds before accessing high_mut_regions[i-1]
                        if i > 0 and len(high_mut_regions) > i:
                            if (
                                high_mut_regions[i - 1][2] >= high_mut_regions[i][1]
                                and high_mut_regions[i - 1][0] == high_mut_regions[i][0]
                            ):
                                # Might need to research a more accurate way to get the mutation rate for this region
                                avg_mut_rate = (
                                    0.5 * high_mut_regions[i - 1][3]
                                    + 0.5 * high_mut_regions[i][3]
                                )
                                high_mut_regions[i - 1] = (
                                    high_mut_regions[i - 1][0],
                                    high_mut_regions[i - 1][1],
                                    high_mut_regions[i][2],
                                    avg_mut_rate,
                                )
                                del high_mut_regions[i]
                    _LOG.info(
                        f"    High mutation region collapsing loop took {time.time() - collapse_start_time:.4f} seconds"
                    )
        _LOG.info(
            f"  High mutation region identification for {contig} took {time.time() - high_mut_start_time:.4f} seconds"
        )
        _LOG.info(
            f"Processing contig {contig} took {time.time() - contig_start_time:.4f} seconds"
        )

    _LOG.info(
        f"Total contig processing loop took {time.time() - start_time_contig_loop:.4f} seconds"
    )

    # if for some reason we didn't find any valid input variants, exit gracefully...
    total_var = snp_count + sum(indel_count.values())  # Corrected calculation
    if not total_var:
        _LOG.error(
            "Error: No valid variants were found, model could not be created. "
            "Check that names are compatible."
        )
        sys.exit(1)

    # COMPUTE PROBABILITIES
    prob_calc_start_time = time.time()

    # frequency that each trinuc mutated into anything else
    trinuc_mut_prob = {}
    # frequency that a trinuc mutates into another trinuc, given that it mutated
    trinuc_trans_probs = {}
    # frequency of snp transitions, given a snp occurs.
    snp_trans_freq = {}

    trinuc_prob_loop_start = time.time()
    for trinuc in sorted(trinuc_ref_count):
        my_count = 0
        for k in sorted(trinuc_transition_count):
            if k[0] == trinuc:
                my_count += trinuc_transition_count[k]

        # Avoid division by zero if trinuc_ref_count[trinuc] is 0
        if float(trinuc_ref_count[trinuc]) > 0:
            trinuc_mut_prob[trinuc] = my_count / float(trinuc_ref_count[trinuc])
        else:
            trinuc_mut_prob[trinuc] = (
                0.0  # Assign 0 probability if the reference count is 0
            )

        # Avoid division by zero if my_count is 0
        if float(my_count) > 0:
            for k in sorted(trinuc_transition_count):
                if k[0] == trinuc:
                    trinuc_trans_probs[k] = trinuc_transition_count[k] / float(my_count)
        else:
            # If my_count is 0, assign 0 probability to all transitions from this trinuc
            trinuc_mut = [
                trinuc[0] + n + trinuc[2] for n in ALLOWED_NUCL if n != trinuc[1]
            ]
            for trinuc2 in trinuc_mut:
                trinuc_trans_probs[(trinuc, trinuc2)] = 0.0

    _LOG.info(
        f"Trinucleotide probability calculation loop took {time.time() - trinuc_prob_loop_start:.4f} seconds"
    )

    snp_prob_loop_start = time.time()
    for n1 in ALLOWED_NUCL:
        rolling_tot = sum(
            [
                snp_transition_count[(n1, n2)]
                for n2 in ALLOWED_NUCL
                if (n1, n2) in snp_transition_count
            ]
        )
        # Avoid division by zero if rolling_tot is 0
        if float(rolling_tot) > 0:
            for n2 in ALLOWED_NUCL:
                key2 = (n1, n2)
                if key2 in snp_transition_count:
                    snp_trans_freq[key2] = snp_transition_count[key2] / float(
                        rolling_tot
                    )
                # else: # Optionally assign 0 if the transition wasn't observed
                #     snp_trans_freq[key2] = 0.0
        # else: # Optionally assign 0 if no transitions from n1 were observed
        #     for n2 in ALLOWED_NUCL:
        #         snp_trans_freq[(n1, n2)] = 0.0

    _LOG.info(
        f"SNP probability calculation loop took {time.time() - snp_prob_loop_start:.4f} seconds"
    )

    # compute average snp and indel frequencies
    avg_freq_start_time = time.time()
    deletion_counts = {abs(key): val for key, val in indel_count.items() if key < 0}
    insertion_counts = {key: val for key, val in indel_count.items() if key > 0}
    average_snp_freq = snp_count / float(total_var)
    average_insertion_frequency = sum(insertion_counts.values()) / float(total_var)
    average_deletion_frequency = sum(deletion_counts.values()) / float(total_var)
    # Avoid division by zero for insertion/deletion frequencies
    insertion_freqency = {
        key: (
            (val / float(total_var)) / average_insertion_frequency
            if average_insertion_frequency > 0
            else 0.0
        )
        for key, val in insertion_counts.items()
    }
    deletion_frequency = {
        key: (
            (val / float(total_var)) / average_deletion_frequency
            if average_deletion_frequency > 0
            else 0.0
        )
        for key, val in deletion_counts.items()
    }
    if homozygous_count:
        homozygous_frequency = homozygous_count / float(total_var)
    else:
        homozygous_frequency = DEF_HOMOZYGOUS_FRQ

    if bed:
        avg_mut_rate = total_var / bed_track_length if bed_track_length > 0 else 0.0
    else:
        avg_mut_rate = total_var / float(total_reflen) if total_reflen > 0 else 0.0
    _LOG.info(
        f"Average frequency calculations took {time.time() - avg_freq_start_time:.4f} seconds"
    )

    # if values weren't found in data, appropriately append null entries
    fill_missing_start_time = time.time()
    print_trinuc_warning = False
    for trinuc in ALL_TRINUCS:
        trinuc_mut = [trinuc[0] + n + trinuc[2] for n in ALLOWED_NUCL if n != trinuc[1]]
        if trinuc not in trinuc_mut_prob:
            trinuc_mut_prob[trinuc] = 0.0
            print_trinuc_warning = True
        for trinuc2 in trinuc_mut:
            if (trinuc, trinuc2) not in trinuc_trans_probs:
                trinuc_trans_probs[(trinuc, trinuc2)] = 0.0
                print_trinuc_warning = True
    if print_trinuc_warning:
        _LOG.warning(
            "Warning: Some trinucleotides transitions were not encountered in the input dataset, "
            "probabilities of 0.0 have been assigned to these events."
        )
    _LOG.info(
        f"Filling missing trinucleotide probabilities took {time.time() - fill_missing_start_time:.4f} seconds"
    )
    _LOG.info(
        f"Total probability calculations took {time.time() - prob_calc_start_time:.4f} seconds"
    )

    # Display counts, if requested
    if show_trinuc:
        for k in sorted(trinuc_mut_prob):
            _LOG.info(f"p({k} mutates) = {trinuc_mut_prob[k]}")

        for k in sorted(trinuc_trans_probs):
            _LOG.info(
                f"p({k[0]} --> {k[1]} | {k[0]} mutates) = {trinuc_trans_probs[k]}"
            )

        for k in sorted(deletion_counts):
            _LOG.info(
                f"p(del length = {abs(k)} | deletion occurs) = {deletion_frequency[k]}"
            )

        for k in sorted(insertion_counts):
            _LOG.info(
                f"p(insert length = {abs(k)} | insertion occurs) = {insertion_freqency[k]}"
            )

        for k in sorted(snp_trans_freq):
            _LOG.info(f"p({k[0]} --> {k[1]} | SNP occurs) = {snp_trans_freq[k]}")

    _LOG.info(f"p(snp)   = {average_snp_freq}")
    _LOG.info(f"p(insertion) = {average_insertion_frequency}")
    _LOG.info(f"p(deletion) = {average_deletion_frequency}")
    _LOG.info(f"overall average mut rate: {avg_mut_rate}")
    _LOG.info(f"total variants processed: {total_var}")
    _LOG.info(f"homozygous probability: {homozygous_frequency}")

    # Convert data structures for model
    conversion_start_time = time.time()
    variant_probs = {
        SingleNucleotideVariant: average_snp_freq,
        Insertion: average_insertion_frequency,
        Deletion: average_deletion_frequency,
    }

    convert_trinuc_trans_start = time.time()
    trinuc_transition_bias = convert_trinuc_transition_matrix(trinuc_trans_probs)
    _LOG.info(
        f"  convert_trinuc_transition_matrix took {time.time() - convert_trinuc_trans_start:.4f} seconds"
    )

    convert_trinuc_mut_start = time.time()
    trinuc_mutation_probs = convert_trinuc_mutation_dict(trinuc_mut_prob)
    _LOG.info(
        f"  convert_trinuc_mutation_dict took {time.time() - convert_trinuc_mut_start:.4f} seconds"
    )

    convert_snp_trans_start = time.time()
    snp_transition_bias = convert_snp_transition_matrix(snp_trans_freq)
    _LOG.info(
        f"  convert_snp_transition_matrix took {time.time() - convert_snp_trans_start:.4f} seconds"
    )

    _LOG.info(
        f"Data structure conversions took {time.time() - conversion_start_time:.4f} seconds"
    )

    # Create model object
    model_create_start_time = time.time()
    mut_model = MutationModel(
        avg_mut_rate=avg_mut_rate,
        homozygous_freq=homozygous_frequency,
        variant_probs=variant_probs,
        transition_matrix=snp_transition_bias,
        trinuc_trans_matrices=trinuc_transition_bias,
        trinuc_mut_bias=trinuc_mutation_probs,
        insert_len_model=insertion_freqency,
        deletion_len_model=deletion_frequency,
    )
    _LOG.info(
        f"MutationModel object creation took {time.time() - model_create_start_time:.4f} seconds"
    )

    print("\nSaving model...")
    save_start_time = time.time()
    with open_output(output, "wb") as outfile:  # Use 'wb' for pickle binary writing
        pickle.dump(mut_model, outfile)
    _LOG.info(f"Saving model took {time.time() - save_start_time:.4f} seconds")
    _LOG.info(
        f"Total runner execution time: {time.time() - start_time_total:.4f} seconds"
    )


def compute_mut_runner(
    reference,
    mutations,
    bed,
    outcounts,
    show_trinuc,
    save_trinuc,
    human_sample,
    skip_common,
    output,
    overwrite_output,
    cores,  # Pass cores argument
):
    """
    Runner for computing the mutation model

    :param str reference: The path to the FASTA reference to use for the analysis
    :param str mutations: The path to the VCF file to use for the analysis
    :param str bed: Optional bed file path for focusing the analysis
    :param str outcounts: Optional path to trinucleotide counts input file
    :param bool show_trinuc: Optionally display trinucleotide counts
    :param bool save_trinuc: Optionally save trinucleotide counts
    :param bool human_sample: If true, treat as human sample
    :param bool skip_common: if true, skip common variants
    :param str output: path to output file
    :param bool overwrite_output: True to overwrite output
    :param int cores: Number of cores to use for processing
    """
    start_time_compute_runner = time.time()

    validate_input_path(reference)
    validate_input_path(mutations)
    mutations = Path(mutations)

    if bed:
        validate_input_path(bed)
        bed = Path(bed)

    if outcounts:
        validate_input_path(outcounts)
        outcounts = Path(outcounts)
    elif save_trinuc:
        # Ensure output is a string before concatenation
        output_str = str(output) if isinstance(output, Path) else output
        outcounts = Path(output_str + ".trinuc.pickle.gz")

    print("Processing reference...")
    ref_index_start = time.time()
    reference_index = SeqIO.index(reference, "fasta")
    _LOG.info(f"Reference indexing took {time.time() - ref_index_start:.4f} seconds")

    extract_header_start = time.time()
    vcf_header = extract_header(mutations)
    _LOG.info(f"extract_header took {time.time() - extract_header_start:.4f} seconds")
    vcf_columns = vcf_header[-1]

    vcf_processing_start = time.time()
    if bed:
        vcf_columns = ["bed_chr", "bed_pos1", "bed_pos2"] + vcf_columns
        _LOG.info("Intersecting bed and vcf.")

        # create a dictionary to store the bed ranges
        bed_ranges = {}

        with open(bed, "r") as bed_file:
            for line in bed_file:
                parts = line.strip().split("\t")
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])

                if len(parts) > 3:
                    mut_rate = parts[3]
                else:
                    mut_rate = "."

                if chrom not in bed_ranges:
                    bed_ranges[chrom] = []

                bed_ranges[chrom].append((start, end, mut_rate))

        # make a temporary VCF file for processing
        temp_vcf_lines = []
        temp_vcf_creation_start = time.time()

        with open(mutations, "r") as vcf_file:
            for line in vcf_file:

                if line.startswith("#CHROM"):
                    temp_vcf_lines.append(
                        line.rstrip() + "\tMUTATION_RATES\n"
                    )  # add column

                elif line.startswith("#"):
                    temp_vcf_lines.append(line)  # preserve header lines

                else:
                    parts = line.strip().split("\t")
                    chrom = parts[0]
                    pos = int(parts[1])

                    if chrom in bed_ranges:
                        # check if the VCF record position is within any of the bed ranges
                        for start, end, mut_rate in bed_ranges[chrom]:

                            if start <= pos <= end:
                                parts.append(mut_rate)
                                temp_vcf_lines.append("\t".join(parts) + "\n")
                                break

        # write the selected VCF lines to the temporary file
        with open("temp.vcf", "w") as temp_vcf_file:
            temp_vcf_file.writelines(temp_vcf_lines)
        _LOG.info(
            f"Temp VCF creation took {time.time() - temp_vcf_creation_start:.4f} seconds"
        )

        _LOG.info("Created temp vcf for processing.")

        # set vcf_to_process to the path of the temporary VCF file
        vcf_to_process = pathlib.Path("temp.vcf")

    else:
        vcf_to_process = mutations
    _LOG.info(
        f"VCF pre-processing (BED intersection/setup) took {time.time() - vcf_processing_start:.4f} seconds"
    )

    # Ensure output is a string before concatenation
    output_str = str(output) if isinstance(output, Path) else output
    output = Path(output_str + ".pickle.gz")
    validate_output_path(output, overwrite=overwrite_output)

    runner_call_start = time.time()
    runner(
        reference_index,
        vcf_to_process,
        vcf_columns,
        outcounts,
        show_trinuc,
        save_trinuc,
        output,
        bed,
        human_sample,
        skip_common,
        cores,  # Pass cores
    )
    _LOG.info(
        f"Main runner function call took {time.time() - runner_call_start:.4f} seconds"
    )

    if os.path.exists("temp.vcf"):
        os.remove("temp.vcf")

    _LOG.info(f"Complete! Use {output} as input into gen_reads_runner.py.")
    _LOG.info(
        f"Total compute_mut_runner execution time: {time.time() - start_time_compute_runner:.4f} seconds"
    )
