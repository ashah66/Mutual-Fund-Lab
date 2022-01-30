"""
Compare row data with the immediate row data of index plus one.  Features are calculated
including:

 (1) Number of matching sentences between rows
 (2) Matching language between rows
 (3) Unique language between rows
 (4) Calculated jaccard similarity score
"""
import numpy as np
import pandas as pd


def run_sentence_compare(data, column):
    """
    Runs all required functions to calculate text features

    Args:
        data: text dataframe
        column: string identifier for column selection

    Returns:
        data: return dataframe of calculated text features
    """

    (num_matching,
     recycled_language,
     unique_language,
     jaccard) = _compare_sentences(data, column)

    data['number_matching'] = num_matching
    data['recycled_language'] = recycled_language
    data['unique_language'] = unique_language
    data['jaccard'] = jaccard

    data['number_sentences'] = [len(sent) for sent in data[column]]
    data['percent_matching'] = data['number_matching'] / data['number_sentences']

    return data


def _compare_sentences(data, column):
    """
    Compares number of matching sentences by single descending row

    Args:
        data: text dataframe
        column: string identifier for column selection

    Returns:
        num_matching: Series of calculated values
        recycled_language: Series of calculated values
        unique_language: Series of calculated values
        jaccard: Series of calculated values
    """
    # Initialize lists and first row value which not be tracked
    data['compare_accession#'] = data['accession#'].shift(1)
    compare_row = [0]
    num_matching = [0]
    recycled_language = [0]
    unique_language = [0]
    jaccard = [0]

    for row, _ in data.iterrows():
        try:
            if data['fund_name'][row] != data['fund_name'][row + 1]:
                data['compare_accession#'].iloc[row + 1] = 0

                compare_row.append(0)

                num_matching.append(0)

                recycled_language.append(0)

                unique_language.append(0)

                jaccard.append(0)

            else:
                compare_row.append(data[column][row])

                num_matching.append(len(set(data[column][row]) & set(data[column][row + 1])))

                recycled_language.append(set(data[column][row]) & set(data[column][row + 1]))

                unique_language.append(set(data[column][row+1]).difference(set(data[column][row])))

                jaccard.append(_calculate_jaccard_similarity(data[column][row],
                                                             data[column][row+1]))
        except KeyError:
            pass

    return num_matching, recycled_language, unique_language, jaccard


def _calculate_jaccard_similarity(sentence1, sentence2):
    """
    The Jaccard similarity index measures the similarity between two sets of data. It can range
    from 0 to 1. The higher the number, the more similar the two sets of data.

    Args:
        sentence1: a tokenized sentence
        sentence2: a tokenized sentence

    Returns:
        similarity: score value from 0 to 1
    """

    doc_1_clean = set(sentence1)
    doc_2_clean = set(sentence2)

    intersection = doc_1_clean.intersection(doc_2_clean)
    union = doc_1_clean.union(doc_2_clean)

    similarity = float(len(intersection) / len(union))

    return similarity
