"""
Class for reads. Each read is described by a name, start position, end position, quality array, mutations, errors,
and whether it is on the reverse strand.

In addition, we attach the reference sequence for later retrieval.

Methods allow comparisons between reads, based on chromosome, start and end. Also, there are methods to retrieve
both the reference sequence and the read and the actual read sequence.
"""
import logging
import numpy as np

from typing import TextIO
from Bio.Seq import Seq, MutableSeq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from numpy.random import Generator

from ...common import ALLOWED_NUCL
from ...models import SequencingErrorModel, ErrorContainer, MutationModel, TraditionalQualityModel
from ...variants import SingleNucleotideVariant, Insertion, Deletion

_LOG = logging.getLogger(__name__)


class Read:
    """
    A class representing a particular read

    :param name: name for the read
    :param raw_read: gives the read as a tuple of coordinates, e.g., (1, 110, 111, 125)
        Which gives information about the left and paired right read, or gives 0,0 for missing component
    :param reference_segment: The reference segment this read is drawn from
    :param reference_id: the ID for the reference where the segment is drawn from
    :param position: First position of the read, relative to the reference
    :param end_point: End point of the read, relative to the reference
    :param padding: The amount of padding we have for this read, will influence if we can add deletions
    :param run_read_len: The final read length desired.
    :param is_reverse: Whether the read is reversed
    :param is_paired: Whether this read has a proper pair.
    """
    def __init__(self,
                 name: str,
                 raw_read: tuple,
                 reference_segment: Seq,
                 reference_id: str,
                 position: int,
                 end_point: int,
                 padding: int,
                 run_read_len: int,
                 is_reverse: bool = False,
                 is_paired: bool = False):

        self.name = name
        self.raw_read = raw_read
        # This segment will gold the raw read and be untouched.
        self.reference_segment = reference_segment
        self.reference_id = reference_id
        self.position = position
        self.end_point = end_point
        self.padding = padding
        self.run_read_length = run_read_len
        self.is_reverse = is_reverse
        self.is_paired = is_paired

        # These parameters won't be initialized on the initial read creation, but will
        # be generated by functions below.
        self.read_sequence: Seq = Seq("")  # initialize to empty sequence
        self.errors: list[ErrorContainer] = []  # initialize
        self.mutations: dict[int: list] = {}  # initialize
        self.quality_array: np.ndarray = np.zeros(self.run_read_length)  # this will have the correct memory length
        self.mapping_quality: int = 0  # initialize at 0
        self.read_quality_string: str = ""  # This will hold the read quality string
        self.num_ns = 0

    def __repr__(self):
        return f"{self.reference_id}: {self.position}-{self.end_point}"

    def __str__(self):
        return f"{self.reference_id}: {self.position}-{self.end_point}"

    def __gt__(self, other):
        if self.reference_id == other.reference_id:
            return self.position > other.position
        else:
            return False

    def __ge__(self, other):
        if self.reference_id == other.reference_id:
            return self.position >= other.position
        else:
            return False

    def __lt__(self, other):
        if self.reference_id == other.reference_id:
            return self.position < other.position
        else:
            return False

    def __le__(self, other):
        if self.reference_id == other.reference_id:
            return self.position <= other.position
        else:
            return False

    def __ne__(self, other):
        if self.reference_id == other.reference_id:
            return self.position != other.position or self.end_point != other.end_point
        else:
            return True

    def __eq__(self, other):
        if self.reference_id == other.reference_id:
            return self.position == other.position and self.end_point == other.end_point
        else:
            return False

    def __len__(self):
        return self.run_read_length

    def update_quality_array(
            self,
            ref_length: int,
            alternate: Seq,
            location: int,
            variant_type: str,
            quality_scores: list,
            quality_score: int = 0
    ):
        """
        This updates the quality score based on the error model. Uniform for mutations, random (but low) for errors
        :param ref_length: The length of the reference sequence for the variant
        :param alternate: The alternate sequence for the variant
        :param location: The first position of the mutation, in 0-based coordinates
        :param variant_type: Either "mutations" or "errors"
        :param quality_scores: The possible quality scores, used to adjust for mutations and errors
        :param quality_score: The quality score to use, since this has already been calculated for mutations.
            This must be included if type is 'mutation'
        :return: None, updates the quality array in place
        """
        if variant_type == "mutation":
            new_quality_score = [quality_score] * len(alternate)
        else:
            # Since we have an error here, we'll choose a min score
            low_score = min(quality_scores)
            # If is insertion
            if len(alternate) > 1:
                # Generate extra scores for insertions
                # Original ref is unaffected, so it's quality score remains the same
                new_quality_score = [low_score] * (len(alternate) - 1)
            # If is deletion
            elif ref_length > 1 and len(alternate) == 1:
                new_quality_score = []
            # SNP
            else:
                new_quality_score = [low_score]

        # Replace the given quality score with the new one
        self.quality_array = \
            np.concatenate((self.quality_array[:location],
                            np.array(new_quality_score),
                            self.quality_array[location+ref_length:]))

    def apply_errors(self, quality_model: TraditionalQualityModel):
        """
        This function applies errors to a sequence and calls the update_quality_array function after

        :param quality_model: The error model for this run,
        :return: None, The sequence, with errors applied
        """
        mutated_sequence = MutableSeq(self.read_sequence)
        for error in self.errors:
            # Replace the entire ref sequence with the entire alt sequence
            mutated_sequence = \
                mutated_sequence[:error.location] + error.alt + mutated_sequence[error.location+len(error.ref):]
            # update quality score for error
            self.update_quality_array(
                len(error.ref),
                error.alt,
                error.location,
                "error",
                list(quality_model.quality_scores),
            )

        self.read_sequence = Seq(mutated_sequence)

    def apply_mutations(self, quality_scores: list, rng: Generator):
        """
        Applying mutations involves one extra step, because of polyploidism. There may be more than one mutation
        at a given location, so it is formulated as a list. We then pick one at random for this read.

        :param quality_scores: The possible quality scores for this run (to update quality scores)
        :param rng: the random number generator for the run
        :return: mutated sequence, with mutations applied
        """
        mutated_sequence = MutableSeq(self.read_sequence)
        for location in self.mutations:
            variant_to_apply = rng.choice(self.mutations[location])

            # Fetch parameters
            qual_score = variant_to_apply.get_qual_score()
            # Find the position within the read for this variant, cast as a python int, instead of a numpy int
            position = int(variant_to_apply.get_0_location() - self.position)
            # Figure out if the variant is in this read. If 'to_mutate' selects any 1, then it is mutated.
            # For diploid animals, for example, this should result in approximately 50% of reads having a
            # given heterozygous mutation and 100% for homozygous mutations.
            to_mutate = rng.choice(variant_to_apply.genotype)
            # If a 1 was selected, then apply the variant, else return the original sequence
            if to_mutate:
                if type(variant_to_apply) == Insertion or type(variant_to_apply) == SingleNucleotideVariant:
                    reference_length = 1
                    alternate = variant_to_apply.get_alt()
                elif type(variant_to_apply) == Deletion:
                    reference_length = variant_to_apply.length
                    if self.padding - variant_to_apply.length < 0:
                        # Skip this deletion, as there is insufficient space
                        self.padding = 0
                        continue
                    else:
                        self.padding -= variant_to_apply.length
                    alternate = self.read_sequence[position]
                else:
                    reference_length = variant_to_apply.get_ref_len()
                    alternate = variant_to_apply.get_alt()

                # Replace the entire ref with the entire alt
                mutated_sequence = \
                    mutated_sequence[:position] + \
                    alternate + \
                    mutated_sequence[position + reference_length:]

                self.update_quality_array(
                    reference_length,
                    alternate,
                    location,
                    "mutation",
                    quality_scores,
                    qual_score
                )

        # Update the read sequence with the applied mutations
        self.read_sequence = Seq(mutated_sequence)

    def apply_variants_for_final_output(
            self,
            quality_model: TraditionalQualityModel,
            rng: Generator
    ):
        """
        Gets mutated sequence to output for fastq/bam

        :param quality_model: The error model for the run
        :param rng: the random number generator for the run
        :return: the mutated sequence
        """
        # I realize that it's a little weird to convert self.read_sequence to mutable then assign to a variable
        # instead of operating on it in place, but while it is mutable, it was easier to treat it like a variable then
        # reconvert and reassign.
        if self.mutations:
            self.apply_mutations(list(quality_model.quality_scores), rng)
        if self.errors:
            self.apply_errors(quality_model)

        # The segments we hold off truncating because we need them for alignments,
        # but the quality array is fine to trim here.
        self.quality_array = self.quality_array[:self.run_read_length]

    def contains(self, test_pos: int):
        return self.position <= test_pos < self.end_point

    def calculate_flags(self, paired_ended_run):
        """
        Calculates the flags for the read

        :param paired_ended_run: Whether the entire run was done in paired-ended mode
        """
        flag = 0
        if paired_ended_run:
            flag += 1
            if self.is_paired:
                # Whether this read has a mapped pair (proper pair)
                flag += 2
            # Flag 4 is if this read is unmapped, which isn't a situation we'll encounter in the simulation
            if not self.is_paired:
                # If this read's mate is unmapped (usually because it was off the end)
                flag += 8
            if self.is_reverse:
                # Flag to indicate this is the reverse strand
                flag += 16
            elif self.is_paired:
                # Flag that indicates that the mate is reversed
                # (which is always the case, if it exists, in this simulation)
                flag += 32
            if not self.is_reverse and self.is_paired:
                flag += 64
            if self.is_reverse and self.is_paired:
                flag += 128
            # None of the other potential samflags are relevant to this simulation
        return flag

    def finalize_read_and_write(
            self,
            err_model: SequencingErrorModel,
            qual_model: TraditionalQualityModel,
            fastq_handle: TextIO,
            quality_offset: int,
            produce_fastq: bool,
            rng: Generator
    ):
        """
        Writes the record to the temporary fastq file

        :param err_model: The error model for the run
        :param qual_model: The quality score model for the run
        :param fastq_handle: the path to the fastq model to write the read
        :param quality_offset: the quality offset for this run
        :param produce_fastq: If true, this will write out the temp fastqs. If false, this will only write out the tsams
            to create the bam files.
        :param rng: the random number generator for this run
        """

        # Generate quality scores for the read
        self.quality_array = qual_model.get_quality_scores(err_model.read_length, len(self.reference_segment), rng)

        # This replaces either hard or soft-masked reference segment with upper case or a standard repeat
        # It updates the quality array and reference segment in place, including reversing them, if appropriate
        self.convert_masking(qual_model)

        # set the read sequence to match the reference
        self.read_sequence = self.reference_segment

        # Get errors for the rea and update the quality score
        self.errors, self.padding = err_model.get_sequencing_errors(
            self.padding,
            self.reference_segment,
            self.quality_array,
            rng
        )

        # This applies any variants, updates quality score and read sequence in place
        self.apply_variants_for_final_output(qual_model, rng)

        self.read_quality_string = "".join([chr(int(x) + quality_offset) for x in self.quality_array])
        # If this read isn't low quality, pick a standard mapping quality
        # We could have this be user assigned.
        if not self.mapping_quality:
            self.mapping_quality = 70

        if produce_fastq:
            fastq_handle.write(f'@{self.name}\n')
            fastq_handle.write(f'{str(self.read_sequence[:self.run_read_length])}\n')
            fastq_handle.write('+\n')
            fastq_handle.write(f'{self.read_quality_string}\n')

    def convert_masking(self, quality_model: TraditionalQualityModel):
        """
        Replaces invalid characters with random valid characters drawn from a standard repeat sequence seen in a lot
        of different species (TTAGGG). If there is call for it, we can make this customizable to species.

        :param quality_model: The error model for this run
        :return: The modified sequence object
        """
        bad_score = min(quality_model.quality_scores)
        # we'll use generic human repeats, as commonly found in masked regions. We may refine this to make configurable
        repeat_bases = list("TTAGGG")
        if self.is_reverse:
            raw_sequence = self.reference_segment.reverse_complement().upper()
            self.quality_array = self.quality_array[::-1]
        else:
            raw_sequence = self.reference_segment.upper()

        start = raw_sequence.find('N')
        if start != -1:
            modified_segment = MutableSeq(raw_sequence[:start])
            for i in range(start, len(raw_sequence)):
                base = raw_sequence[i]
                if base in ALLOWED_NUCL:
                    modified_segment += base
                else:
                    modified_segment += repeat_bases[i % 6]
                    self.quality_array[i] = bad_score
        else:
            modified_segment = MutableSeq(raw_sequence)

        self.reference_segment = Seq(modified_segment)

    def align_seqs(self):
        """
        The sequence alignment. We restrict the alignment to the section of the reference where we know the read
        came from and try to generate a minimal cigar string. The cigar string part may still need tweaking.
        """
        raw_alignment = pairwise2.align.globalms(
            self.reference_segment,
            self.read_sequence,
            match=10,
            mismatch=-10,
            open=-20,
            extend=-10,
            penalize_extend_when_opening=True,
            one_alignment_only=True,
        )
        alignment = format_alignment(*raw_alignment[0], full_sequences=True).split()
        aligned_template_seq = alignment[0]
        aligned_mut_seq = alignment[-2]
        cig_count = 0
        cig_length = 0
        curr_char = ''
        cig_string = ''
        # Find first match. Added a +1 because all my matches were coming up short.
        for char in range(len(self.read_sequence) + 1):
            if aligned_template_seq[char] == '-':  # insertion
                if curr_char == 'I':  # more insertions
                    cig_count += 1
                    cig_length += 1
                else:  # new insertion
                    cig_string = cig_string + str(cig_count) + curr_char
                    curr_char = 'I'
                    cig_count = 1
                    cig_length += 1
            elif aligned_mut_seq[char] == '-':  # deletion
                if curr_char == 'D':  # more deletions
                    cig_count += 1
                else:  # new deletion
                    if cig_count != 0:
                        cig_string = cig_string + str(cig_count) + curr_char
                    curr_char = 'D'
                    cig_count = 1
            else:  # match
                if curr_char == 'M':  # more matches
                    cig_count += 1
                    cig_length += 1
                else:  # new match
                    # If there is anything before this, add it to the string and increment,
                    # else, just increment
                    if cig_count != 0:
                        cig_string = cig_string + str(cig_count) + curr_char
                    curr_char = 'M'
                    cig_count = 1
                    cig_length += 1
            if cig_length == self.run_read_length:
                break

        return cig_string, cig_count, curr_char, cig_length

    def make_cigar(self):
        """
        Aligns the reference and mutated sequences.
        """
        # These parameters were set to minimize breaks in the mutated sequence and find the best
        # alignment from there.

        cig_string, cig_count, curr_char, cig_length = self.align_seqs()
        if cig_length < self.run_read_length:
            _LOG.warning("Poor alignment, trying reversed")
            cig_string2, cig_count2, curr_char2, cig_length2 = self.align_seqs()
            if cig_length2 < cig_length:
                cig_length = cig_length2
            while cig_length < self.run_read_length:
                cig_string += "M"
                cig_length += 1

        # append the final section as we return
        return cig_string + str(cig_count) + curr_char

    def get_mpos(self):
        """
        Get the mate position of the read
        """
        if self.is_paired:
            if self.is_reverse:
                return self.raw_read[0]
            else:
                return self.raw_read[2]
        else:
            return 0

    def get_tlen(self):
        """
        Get the template length for the read
        """
        if self.is_paired:
            length = self.raw_read[3] - self.raw_read[0] + 1
            if length < 0:
                return 0
            else:
                if self.is_reverse:
                    return -length
                else:
                    return length
        else:
            return 0
