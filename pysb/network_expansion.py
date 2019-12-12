import itertools
import math

import numpy as np
import sympy as sp
import networkx as nx

from pysb import (
    ReactionPattern, ComplexPattern, Monomer, Expression, Observable
)
from pysb.pattern import match_complex_pattern
from networkx.algorithms.isomorphism.vf2userfunc import GraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
from collections import ChainMap, Counter


def partial_network_expansion(model, seeds):
    reactions = []
    rule_matches = [
        {
            'name': f'{rule.name}{"__reverse" if reverse else ""}',
            'rule': rule.name,
            'reverse': reverse,
            'reactant_pattern':
                rule.product_pattern if reverse
                else rule.reactant_pattern,
            'product_pattern':
                rule.reactant_pattern if reverse
                else rule.product_pattern,
            'rate':
                None if rule.energy
                else rule.rate_reverse if reverse
                else rule.rate_forward,
            'energy': rule.energy,
            'phi':
                rule.rate_forward if rule.energy
                else None,
            'Ea0':
                rule.rate_reverse if rule.energy
                else None,
            'matches':
                np.zeros((len(rule.product_pattern.complex_patterns), 0))
                if reverse
                else np.zeros((len(rule.reactant_pattern.complex_patterns), 0))
        }
        for rule in model.rules
        for reverse in [True, False]
        if reverse == rule.is_reversible or not reverse
    ]
    species = seeds
    index_added_species = list(range(len(seeds)))
    index_updated_species = index_added_species
    iterations = 0
    print('Partial Network Expansion')
    while len(index_updated_species):
        print(f'Iteration {iterations}:')
        for rule in rule_matches:
            if len(rule['reactant_pattern'].complex_patterns) == 0:
                continue
            # extend match matrix to added species
            rule['matches'] = np.concatenate(
                (
                    rule['matches'],
                    get_matching_patterns(
                        rule['reactant_pattern'],
                        [species[i] for i in index_added_species]
                    )
                ), axis=1
            )
        index_old_species = [i for i in range(len(species))
                             if i not in index_updated_species]
        new_reactions = []
        for rule in rule_matches:
            rule_reactions = []
            if rule['matches'].shape[0] == 0:
                continue

            old_matches = rule['matches'].copy()
            old_matches[:, index_updated_species] = False
            new_matches = rule['matches'].copy()
            new_matches[:, index_old_species] = False

            # here we need to account for the full combinatorial space of
            # possible combinations of new and old matches. We loop over the
            # number of new matches and then use itertools to select all
            # possible combinations
            for new_match_count in range(1, sum(new_matches.any(axis=1))+1):
                for new_match_sel in itertools.combinations(
                    np.where(new_matches.any(axis=1))[0],
                    r=new_match_count
                ):
                    # if educt index icol is in ie, we select all pattern
                    # matches from newly generated species, otherwise,
                    # we select a pattern matches from old
                    educt_matches = [
                        np.where(new_matches[irp, :])[0]
                        if irp in new_match_sel
                        else np.where(old_matches[irp, :])[0]
                        for irp in range(rule['matches'].shape[0])
                    ]
                    # generate the reactions for the respective matches
                    rule_reactions.extend(
                        get_reactions_from_matches(
                            educt_matches, rule, model.energypatterns,
                            species
                        )
                    )

            if len(rule_reactions):
                print(f'Rule {rule["name"]}: {len(rule_reactions)} new '
                      f'reactions')
            new_reactions.extend(rule_reactions)

        index_updated_species = []
        index_added_species = []
        # loop over all the reactions
        while len(new_reactions):
            reaction = new_reactions.pop(0)
            products = []
            # for every produces species, check if it is already present in
            # the list of species
            for specie_canditate in reaction['product_patterns']:
                species_index = next(
                    (
                        ispecie
                        for ispecie, specie in enumerate(species)
                        if match_complex_pattern(
                            specie, specie_canditate,
                        )
                    ),
                    None
                )
                if species_index is None:
                    # no match was found in existing species, add the species
                    # and update added/updated species
                    species.append(specie_canditate)
                    index_added_species.append(len(species)-1)
                    index_updated_species.append(len(species)-1)
                    # store indexing for easy future access
                    products.append(len(species)-1)
                else:
                    products.append(species_index)
                    # check if the new pattern is more specific
                    if not match_complex_pattern(
                            specie_canditate, species[species_index]
                    ):
                        # replace old pattern by more specific pattern
                        species[species_index] = specie_canditate
                        index_updated_species.append(species_index)

                        for rule in rule_matches:
                            if rule['matches'].shape[0] == 0:
                                # if the match matrix wasnt created yet, we
                                # dont have to do anything
                                continue
                            if rule['matches'].shape[1] < species_index:
                                # if we are replacing a newly added species,
                                # we don't have to update the match matrix yet
                                continue
                            # update the matches
                            rule['matches'][:, species_index] = \
                                get_matching_patterns(
                                    rule['reactant_pattern'],
                                    [species[species_index]]
                                )[:, 0]

                        # remove reactions that used the less specific species,
                        # we will regenerate more specific implementations
                        # of these reactions in the next iteration
                        for ir, r in reversed(list(enumerate(reactions))):
                            if species_index in list(
                                    r['educts'] + r['products']
                            ):
                                del reactions[ir]

                        for ir, r in reversed(list(enumerate(new_reactions))):
                            if species_index in list(r['educts']):
                                del new_reactions[ir]

            reaction['products'] = tuple(products)
            reactions.append(reaction)

        if len(index_updated_species) == 0:
            print('Expansion complete')
        iterations += 1

    return species, reactions


def get_matching_patterns(reaction_pattern, species):
    return np.asarray([
            [
                match_complex_pattern(cp, s)
                if s is not None and cp is not None
                else False
                for s in species
            ]
            for cp in reaction_pattern.complex_patterns
    ])


def get_reactions_from_matches(educt_matches, rule, energy_patterns, species):
    return [
        apply_conversion_to_complex_patterns(
            rule, energy_patterns,
            educt_indices, species
        )
        for educt_indices in itertools.product(*educt_matches)
    ]


def apply_conversion_to_complex_patterns(rule, energy_patterns,
                                         educt_indices, species):
    complexpatterns = [species[e] for e in educt_indices]

    stat_factor = 1.0
    for count in Counter(educt_indices).values():
        stat_factor *= 1 / math.factorial(count)

    mp_alignment_rp, mp_alignment_pp = align_monomer_indices(
        rule['reactant_pattern'], rule['product_pattern']
    )

    rp_graph = rule['reactant_pattern']._as_graph(
        prefix='rp'
    )
    pp_graph = rule['product_pattern']._as_graph(
        prefix='rp', mp_alignment=mp_alignment_pp
    )

    graph_diff = dict()
    graph_diff['removed_nodes'] = [
        n for n, d in rp_graph.nodes(data=True)
        if n not in pp_graph
        or n in pp_graph and pp_graph.nodes[n]['id'] != d['id']
    ]
    rp_graph.remove_nodes_from(graph_diff['removed_nodes'])
    graph_diff['added_nodes'] = [
        (n, d) for n, d in pp_graph.nodes(data=True)
        if n not in rp_graph
    ]
    rp_graph.add_nodes_from(graph_diff['added_nodes'])
    graph_diff['removed_edges'] = nx.difference(rp_graph, pp_graph).edges()
    graph_diff['added_edges'] = nx.difference(pp_graph, rp_graph).edges()

    node_matcher = categorical_node_match('id', default=None)

    def autoinc():
        i = 0
        while True:
            yield i
            i += 1

    # alignment of mps in cps of pattern allows merging of mappings through
    # ChainMap, also enables us to apply the graph diff to the graph of the
    # reactant pattern of all cps in pattern in the end
    mp_count_pattern = autoinc()
    mp_alignment_cp = [
        [next(mp_count_pattern) for _ in cp.monomer_patterns]
        for cp in complexpatterns
    ]

    matches = [
        GraphMatcher(
            cp._as_graph(mp_alignment_cp[icp]),
            rp._as_graph(mp_alignment_rp[icp], prefix='rp'),
            node_match=node_matcher
        )
        for icp, (rp, cp)
        in enumerate(zip(rule['reactant_pattern'].complex_patterns,
                         complexpatterns))
    ]
    reactions = []

    for rpmatch in matches:
        assert(rpmatch.subgraph_is_isomorphic())

    # invert and merge mapping
    full_mapping = dict(ChainMap(*[
        dict(zip(match.mapping.values(), match.mapping.keys()))
        for match in matches
    ]))
    mapped_diff = map_graph_diff(graph_diff, full_mapping)

    educt_graph = ReactionPattern(complexpatterns)._as_graph(
        mp_alignment_cp
    )
    products = reaction_pattern_from_graph(
        apply_graph_diff(educt_graph, mapped_diff)
    ).complex_patterns

    if rule['energy']:
        energy_diffs = [
            {
                'name': name,
                'count': sum(
                    match_complex_pattern(ep.pattern, pattern, count=True) /
                    match_complex_pattern(ep.pattern, ep.pattern, count=True)
                    for pattern in products

                ) - sum(
                    match_complex_pattern(ep.pattern, pattern, count=True) /
                    match_complex_pattern(ep.pattern, ep.pattern, count=True)
                    for pattern in complexpatterns
                ),
                'energy': ep.energy
            }
            for name, ep in energy_patterns.items()
        ]
        energies = set(
            diff['name'] for diff in energy_diffs
            if diff['count'] != 0.0
        )
        deltadeltaG = sum(
            diff['energy']*diff['count'] for diff in energy_diffs
            if diff['count'] != 0.0
        )
        if rule['reverse']:
            rate = sp.exp(-rule['Ea0'] + rule['phi'] * deltadeltaG)
        else:
            rate = sp.exp(-rule['Ea0'] + (1 - rule['phi']) * deltadeltaG)

        subs = []
        for a in rate.atoms():
            if isinstance(a, Expression):
                subs.append((a, a.expand_expr(
                    expand_observables=True)))
            elif isinstance(a, Observable):
                subs.append((a, a.expand_obs()))
        rate = rate.subs(subs)
        rate = sp.powdenest(sp.logcombine(rate, force=True), force=True)
    else:
        energies = set()
        rate = rule['rate']

    reaction = {
        'rule': rule['name'],
        'product_patterns': products,
        'educt_patterns': complexpatterns,
        'rate': stat_factor*rate,
        'energies': energies,
        'educts': tuple(educt_indices),
    }
    assert (len(complexpatterns) ==
            len(rule['reactant_pattern'].complex_patterns))
    assert (len(products) ==
            len(rule['product_pattern'].complex_patterns))

    return reaction


def map_graph_diff(diff, mapping):
    mapped_diff = dict()

    for edges in ['removed_edges', 'added_edges']:
        mapped_diff[edges] = [
            (mapping.get(e[0], e[0]), mapping.get(e[1], e[1]))
            for e in diff[edges]
        ]

    mapped_diff['removed_nodes'] = [
        mapping[n]
        for n in diff['removed_nodes']
    ]
    mapped_diff['added_nodes'] = [
        (mapping.get(n, n), d)
        for n, d in diff['added_nodes']
    ]
    return mapped_diff


def apply_graph_diff(graph, diff):
    outgraph = graph.copy()
    outgraph.remove_nodes_from(diff['removed_nodes'])
    outgraph.add_nodes_from(diff['added_nodes'])
    outgraph.add_edges_from(diff['added_edges'])
    outgraph.remove_edges_from(diff['removed_edges'])
    return outgraph


def align_monomer_indices(reactantpattern, productpattern):

    def autoinc():
        i = 0
        while True:
            yield i
            i += 1

    mp_count = autoinc()
    rp_alignment = [
        [next(mp_count) for _ in cp.monomer_patterns]
        for cp in reactantpattern.complex_patterns
    ]

    rp_monos = [
        mp.monomer.name
        for cp in reactantpattern.complex_patterns
        for mp in cp.monomer_patterns
    ]

    pp_monos = {
        (icp, imp): mp.monomer.name
        for icp, cp in enumerate(productpattern.complex_patterns)
        for imp, mp in enumerate(cp.monomer_patterns)
    }

    pp_alignment = [
        [np.NaN] * len(cp.monomer_patterns)
        for cp in productpattern.complex_patterns
    ]

    for imono, rp_mono in enumerate(rp_monos):
        # find first MonomerPattern in productpattern with same monomer name
        index = next(
            ((icp, imp) for (icp,imp), pp_mono in pp_monos.items()
            if pp_mono == rp_mono),
            None
        )
        # if we find a match, set alignment index and delete to prevent
        # rematch, else continue
        if index is not None:
            pp_alignment[index[0]][index[1]] = imono
            del pp_monos[index]

    # add alignment for all unmatched MonomerPatterns
    for new_count, index in enumerate(pp_monos.keys()):
        pp_alignment[index[0]][index[1]] = -(new_count+1)

    return rp_alignment, pp_alignment


def reaction_pattern_from_graph(graph):
    return ReactionPattern(
        [complex_pattern_from_graph(graph.subgraph(c)) for c in
         nx.connected_components(graph)]
    )


def complex_pattern_from_graph(graph):
    assert(nx.is_connected(graph))
    bounds = list()
    return ComplexPattern(
        [
            monomer_pattern_from_node(graph, node[0], bounds)
            for node in graph.nodes(data=True)
            if isinstance(node[1]['id'], Monomer)
        ],
        None
    )


def monomer_pattern_from_node(graph, monomer_node, bounds):
    monomer = graph.nodes.data()[monomer_node]['id']
    site_conditions = {
        graph.nodes.data()[site]['id']: site_condition_from_node(
            graph, monomer, site, bounds
        )
        for site in graph.neighbors(monomer_node)
    }
    return monomer(**site_conditions)


def site_condition_from_node(graph, monomer, site_node, bounds):
    state = None
    site = graph.nodes.data()[site_node]['id']
    for condition_node in graph.neighbors(site_node):
        state_candidate = graph.nodes.data()[condition_node]['id']
        if state_candidate == 'NoBond':
            continue
        elif isinstance(state_candidate, Monomer):
            continue
        elif site in monomer.site_states and state_candidate in \
                monomer.site_states[site]:
            state = state_candidate
        else:
            if site_node in bounds:
                state = bounds.index(site_node)
            else:
                state = len(bounds)
                bounds.append(condition_node)
    return state