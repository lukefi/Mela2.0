import math
from collections.abc import Callable
from typing import Optional
import numpy as np
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.model import ReferenceTree, TreeStratum
from lukefi.metsi.data.vector_model import (
    ReferenceTrees as VectorReferenceTrees,
    TreeStrata as VectorTreeStrata,
)


def calculate_basal_area(tree: ReferenceTree) -> float:
    """ Single reference tree basal area calculation.

    The tree should contain breast height diameter (in cm) and stesm per hectare for the species spesific calculations.

    :param tree: Single ReferenceTree instance with breast height diameter (in cm) and stems per hectare properties.
    :return reference tree basal area in square meters (m^2)
    """
    meters_factor = 0.01
    radius = (tree.breast_height_diameter or 0.0) * 0.5 * meters_factor
    single_basal_area = math.pi * math.pow(radius, 2)
    return (single_basal_area or 0.0) * (tree.stems_per_ha or 0.0)


def generate_diameter_threshold(d1: float, d2: float) -> float:
    """ Threshold value for diameter based comparison of two stratums.

    Threshold will have a value based on relative distance of at most 50% of the distance between d[0] and d[1].
    """
    greater = max((d1, d2))
    lesser = min((d1, d2))
    return greater + (lesser - greater) * (greater / (lesser + greater))


def override_from_diameter(initial_stratum: TreeStratum, candidate_stratum: TreeStratum,
                           reference_tree: ReferenceTree) -> TreeStratum:
    """ Out of given strata, return the stratum for which the mean diameter better matches the reference tree diameter.
    This happens by calculating a threshold value based on which of the stratum diameters
    is greater and comparing the threshold to reference tree diameter.

    :param initial_stratum: Stratum which is assumed as the current match for the reference tree
    :param candidate_stratum: Stratum which is tested for better compatiblity than the initial stratum
    :param reference_tree: The tree for which the supplementing will be done

    :returns: the better matching stratum
    """
    threshold = generate_diameter_threshold(
        initial_stratum.mean_diameter or 0.0,
        candidate_stratum.mean_diameter or 0.0)
    if not threshold or not reference_tree.breast_height_diameter:
        return initial_stratum
    if threshold > reference_tree.breast_height_diameter:
        return candidate_stratum
    return initial_stratum


def find_matching_stratum_by_diameter_lm(
        reference_tree: ReferenceTree,
        strata: Iterable[TreeStratum],
        threshold=3) -> Optional[TreeStratum]:
    """
    Find the stratum that has the closest diameter to the reference tree diameter by factor of difference, where the
    reference tree diameter is between the stratum mean diameter divided by threshold and multiplied by threshold.

    :param reference_tree: candidate reference tree
    :param strata: candidate strata
    :param threshold: threshold factor for diameter bounds
    :return: matching stratum or None if no match is found
    """

    # h.	Jos em. säännöt ei yksiselitteisesti määrää ositetta, valitaan se osite,
    #   jonka keskiläpimitta on lähinnä puun läpimittaa.
    # i.	Jos puun läpimitta on yli kerroin*valitun ositteen keskiläpimitta, puuta ei kohdisteta sille.
    #   R-koodissa kerroin = 3.

    candidate = min(
        strata,
        # R-koodin mukaisesti puuttuva dgm <- 0
        key=lambda stratum: 0 if stratum.mean_diameter is None else abs(
            reference_tree.breast_height_diameter - stratum.mean_diameter),
        default=None
    )
    if candidate is None:
        return None

    candidate_dgm = candidate.mean_diameter if candidate.mean_diameter is not None else reference_tree.breast_height_diameter
    lower = candidate_dgm / threshold
    upper = candidate_dgm * threshold
    if lower <= reference_tree.breast_height_diameter <= upper:
        return candidate
    else:
        return None


def split_list_by_predicate(items: list, predicate: Callable) -> tuple[list, list]:
    """ Splits a list into two lists based on a predicate.

    :param items: list to be split
    :param predicate: Predicate used to split the list
    :return: Tuple of lists, where the first list contains the items that match the predicate and the second list
        contains the items that do not match the predicate.
    """
    matching_items = []
    non_matching_items = []
    for item in items:
        if predicate(item):
            matching_items.append(item)
        else:
            non_matching_items.append(item)
    return matching_items, non_matching_items


def find_strata_by_similar_species(species: TreeSpecies, strata: list[TreeStratum]) -> list[TreeStratum]:
    """
    Find a list of strata which have a similar species to the given species. Out of deciduous trees,
    silver birch is considered most similar to downy birch and vice versa.
    :param species:
    :param strata:
    :return:
    """
    # e.	Jos koivulle ei ole oman puulajin ositetta, kohdistetaan se toisen koivulajin ositteeseen,
    #   jos sellainen on.
    # f.	Jos havupuulle ei ole olemassa oman puulajin ositetta, kohdistetaan se jonkin muun havupuun ositteeseen.
    # g.	Jos lehtipuulle ei ole oman puulajin ositetta eikä koivulle koivuositetta, kohdistetaan se jonkun muun lehtipuun ositteeseen.
    #   Koivuositteeseen ei kuitenkaan kohdisteta muita kuin koivuja.

    candidates = []

    if species.is_deciduous():
        if species == TreeSpecies.DOWNY_BIRCH:
            candidates.extend(filter(lambda s: s.species == TreeSpecies.SILVER_BIRCH, strata))
        elif species == TreeSpecies.SILVER_BIRCH:
            candidates.extend(filter(lambda s: s.species == TreeSpecies.DOWNY_BIRCH, strata))
        if len(candidates) == 0:
            candidates.extend(
                filter(
                    lambda s: s.species.is_deciduous() and s.species not in (
                        TreeSpecies.SILVER_BIRCH,
                        TreeSpecies.DOWNY_BIRCH),
                    strata))
    elif species.is_coniferous():
        candidates.extend(filter(lambda s: s.species.is_coniferous(), strata))

    return candidates


def find_matching_storey_stratum_for_tree(
        tree: ReferenceTree,
        strata: list[TreeStratum],
        diameter_threshold=3) -> Optional[TreeStratum]:
    # a.	Tarkista, että puu on inventoinnissa mitattu (puutyypit vaihtelee inventointien välillä)
    #   ja se on elävä (elävillä puilla puuluokka on numeerinen).
    if not tree.is_measured_type() or not tree.is_living():
        return None

    same_storey_strata = [stratum for stratum in strata if storey_match(stratum, tree)]
    same_species_strata, other_species_strata = split_list_by_predicate(
        same_storey_strata,
        lambda stratum: stratum.species == tree.species)

    # d.	Puu kohdistetaan ensisijaisesti oman puulajin ositteeseen.
    if len(same_species_strata) > 0:
        candidate_strata = same_species_strata
    elif len(other_species_strata) > 0:
        candidate_strata = find_strata_by_similar_species(tree.species, other_species_strata)
    else:
        candidate_strata = []

    # h.	Jos em. säännöt ei yksiselitteisesti määrää ositetta, valitaan se osite,
    #       jonka keskiläpimitta on lähinnä puun läpimittaa.
    # i.	Jos puun läpimitta on yli kerroin*valitun ositteen keskiläpimitta, puuta ei kohdisteta sille.
    #       R-koodissa kerroin = 3.

    if len(candidate_strata) > 0:
        strata_with_diameter = filter(lambda stratum: stratum.has_diameter(), candidate_strata)
        selected_stratum = find_matching_stratum_by_diameter_lm(tree, candidate_strata, diameter_threshold)
    else:
        selected_stratum = None

    return selected_stratum


def storey_match(stratum: TreeStratum, tree: ReferenceTree):
    # b.	Puu voidaan kohdistaa vain ositteeseen jonka jaksotieto vastaa puun latvuskerrostietoa.
    # c.	Jättöpuu (latvuskerroskoodi kirjain) voidaan kohdistaa vain jättöylispuujaksoon
    #   (koodit F ja G), ja jättöpuujaksoon voidaan kohdistaa vain jättöpuita.
    #   Vallitsevan jakson ja alikasvoksen jättöpuut jäävät aina kohdentamatta, koska
    #   VMI-ohje ei tunne vallitsevan jakson ja alikasvoksen jättöpuujaksoja.

    # |puu$latker=="E") { # alikasvosjakso, jättöaliksavospuita ei kohdisteta millekään ositteelle
    if tree.latvuskerros == "5":
        return stratum.asema == 5 or stratum.asema == 6 or stratum.asema == 9
    elif tree.latvuskerros == "6" or tree.latvuskerros == "7":  # ylispuujakso, ei jättöpuu
        return stratum.asema == 2 or stratum.asema == 4
    elif tree.latvuskerros in ("F", "G"):  # jättöpuut vain jättöylispuujaksoon
        return stratum.asema == 3
    elif tree.latvuskerros in ("2", "3", "4"):  # valitseva jakso
        return stratum.asema <= 1
    else:
        return False
