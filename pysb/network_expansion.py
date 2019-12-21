import itertools
import math

import numpy as np
import sympy as sp
import networkx as nx

from pysb import (
    ReactionPattern, ComplexPattern, MonomerPattern, Monomer, Expression,
    Observable, Compartment
)
from pysb.pattern import match_complex_pattern
from pysb.core import NO_BOND
from networkx.algorithms.isomorphism.vf2userfunc import GraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
from collections import ChainMap, Counter


class NetworkExpansion:
    def __init__(self, model):
        self.model = model
        self.reactions = []
        self.new_reactions = []
        self.species = []

        self.initialize_reaction_generators()

        self.index_added_species = set()
        self.index_updated_species = set()
        self.index_old_species = set()
        self.interations = 0

        self.full_expansion = False

    @property
    def nspecies(self):
        return len(self.species)

    def generate(self, seeds):
        self.index_added_species = set(self.add_species(seeds))
        self.index_updated_species = self.index_added_species
        self.iterations = 0
        self.full_expansion = all(seed.is_concrete() for seed in seeds)

        print('Partial Network Expansion')
        while len(self.index_updated_species):
            print(f'Iteration {self.iterations}:')
            reactions = []

            if self.iterations == 0:
                for generator in self.reaction_generators:
                    if generator.is_pure_synthesis_rule:
                        rule_reactions = [
                            generator.generate_reaction([], self.species,
                                                        self.model)
                        ]

                        if len(rule_reactions):
                            print(
                                f'Rule {generator.name}: {len(rule_reactions)}'
                                f' new reactions')
                        reactions.extend(rule_reactions)

            self.update_reaction_generators_added_species()

            for generator in self.reaction_generators:

                rule_reactions = generator.generate_reactions(
                    self.species, self.index_updated_species,
                    self.index_old_species, self.model
                )

                if len(rule_reactions):
                    print(f'Rule {generator.name}: {len(rule_reactions)} new '
                          f'reactions')
                reactions.extend(rule_reactions)

            self.index_updated_species = set()
            self.index_added_species = set()
            # loop over all the reactions
            while len(reactions):
                reaction = reactions.pop(0)
                # for every produces species, check if it is already present in
                # the list of species
                products = self.add_species(reaction['product_patterns'])

                reaction['products'] = tuple(products)
                self.reactions.append(reaction)

            if len(self.index_updated_species) == 0:
                print('Expansion complete')
            self.iterations += 1

    def add_species(self, candidates):
        species_indices = []
        for cand in candidates:
            species_index = next((
                ispecie
                for ispecie, specie in enumerate(self.species)
                if match_complex_pattern(specie, cand,
                                         exact=self.full_expansion)
            ), None)
            if species_index is None:
                # no match was found in existing species, add the species
                # and update added/updated species
                species_indices.append(self.nspecies)
                self.add_specie(cand)
            else:
                species_indices.append(species_index)
                # check if the new pattern is more specific
                if not match_complex_pattern(cand,
                                             self.species[species_index],
                                             exact=self.full_expansion):
                    self.update_specie(cand, species_index)
        return species_indices

    def add_specie(self, specie):
        self.index_added_species.add(self.nspecies)
        self.index_updated_species.add(self.nspecies)
        self.species.append(specie)

    def update_specie(self, specie, specie_index):
        # replace old pattern by more specific pattern
        self.species[specie_index] = specie
        self.index_updated_species.add(specie_index)

        self.update_reaction_generators_updated_species(specie_index)

        # remove reactions that used the less specific species,
        # we will regenerate more specific implementations
        # of these reactions in the next iteration
        for ir, r in reversed(list(enumerate(self.reactions))):
            if specie_index in list(
                    r['educts'] + r['products']
            ):
                del self.reactions[ir]

        for ir, r in reversed(list(enumerate(self.new_reactions))):
            if specie_index in list(r['educts']):
                del self.new_reactions[ir]

    def initialize_reaction_generators(self):
        self.reaction_generators = [
            ReactionGenerator(rule, reverse, self.model.energypatterns)
            for rule in self.model.rules
            for reverse in [True, False]
            if reverse == rule.is_reversible or not reverse
        ]

    def update_reaction_generators_added_species(self):
        for generator in self.reaction_generators:
            generator.add_matches(
                [self.species[i] for i in sorted(self.index_added_species)]
            )

        self.index_old_species |= {
            i for i in range(len(self.species))
            if i not in self.index_updated_species
        }

    def update_reaction_generators_updated_species(self, specie_index):
        if specie_index in self.index_added_species:
            # if we are replacing a newly added species,
            # we don't have to update the match matrix yet
            return
        for generator in self.reaction_generators:
            generator.update_matches(self.species[specie_index], specie_index)


class ReactionGenerator:
    def __init__(self, rule, reverse, energypatterns):
        self.name = f'{rule.name}{"__reverse" if reverse else ""}'
        self.rule = rule.name,
        self.reverse = reverse,
        self.reactant_pattern = rule.product_pattern if reverse else \
            rule.reactant_pattern
        self.product_pattern = rule.reactant_pattern if reverse else \
            rule.product_pattern
        self.phi = None
        self.Ea0 = None
        if reverse:
            self.rate = rule.rate_reverse
            self.matches = np.zeros(
                (len(rule.product_pattern.complex_patterns), 0)
            )
        else:
            self.rate = rule.rate_forward
            self.matches = np.zeros(
                (len(rule.reactant_pattern.complex_patterns), 0)
            )

        if rule.energy:
            self.rate = None
            self.phi = rule.rate_forward
            self.Ea0 = rule.rate_reverse

        self.energypatterns = energypatterns

        self.is_pure_synthesis_rule = \
            len(rule.reactant_pattern.complex_patterns) == 0

        self.delete_molecules = rule.delete_molecules

        self.graph_diff = GraphDiff(self)

    def add_matches(self, species):
        if self.is_pure_synthesis_rule:
            return
        self.matches = np.concatenate(
            (
                self.matches,
                get_matching_patterns(
                    self.reactant_pattern,
                    species
                )
            ), axis=1
        )

    def update_matches(self, specie, specie_index):
        if self.is_pure_synthesis_rule:
            return
        self.matches[:, specie_index] = \
            get_matching_patterns(
                self.reactant_pattern,
                [specie]
            )[:, 0]

    def generate_reactions(self, species, index_updated_species,
                           index_old_species, model):

        rule_reactions = []
        if self.is_pure_synthesis_rule:
            return rule_reactions
        # here we need to account for the full combinatorial space of
        # possible combinations of new and old matches. We loop over the
        # number of new matches and then use itertools to select all
        # possible combinations

        old_matches = self.matches.copy()
        old_matches[:, list(index_updated_species)] = False
        new_matches = self.matches.copy()
        new_matches[:, list(index_old_species)] = False

        for new_match_count in range(
                max(self.matches.shape[0]-sum(old_matches.any(axis=1)), 1),
                sum(new_matches.any(axis=1)) + 1
        ):
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
                    for irp in range(self.matches.shape[0])
                ]

                # filter symmetric matches, we account for that by
                # statfactor when generating rates
                indice_sets = {
                    tuple(np.unique(indices)): indices
                    for indices in itertools.product(*educt_matches)
                }

                # generate the reactions for the respective matches
                rule_reactions.extend(
                    self.generate_reaction(educt_indices, species, model)
                    for educt_indices in indice_sets.values()
                )
        return rule_reactions

    def generate_reaction(self, educt_indices, species, model):
        educts = [species[e] for e in educt_indices]

        stat_factor = 1
        for count in Counter(educt_indices).values():
            if count == 1:
                continue  # avoid conversion to float
            stat_factor *= 1 / math.factorial(count)

        educt_mapping, mp_alignment_cp = self.compute_educt_mapping(educts)
        self.graph_diff.apply_mapping(educt_mapping)

        educt_graph = ReactionPattern(educts)._as_graph(
            mp_alignment_cp
        )

        products = reaction_pattern_from_graph(
            self.graph_diff.apply(educt_graph, self.delete_molecules),
        ).complex_patterns

        rate, energies = self.get_reaction_rate(
            educts, products, model
        )

        reaction = {
            'rule': self.name,
            'product_patterns': products,
            'educt_patterns': educts,
            'rate': stat_factor * rate * np.prod([sp.Symbol(f'__s{ix}')
                                                  for ix in educt_indices]),
            'energies': energies,
            'educts': tuple(educt_indices),
        }
        return reaction

    def get_reaction_rate(self, educts, products, model):
        if self.rate is None:
            energy_diffs = [
                {
                    'name': name,
                    'count': sum(
                        match_complex_pattern(ep.pattern, pattern,
                                              count=True) /
                        match_complex_pattern(ep.pattern, ep.pattern,
                                              count=True)
                        for pattern in products

                    ) - sum(
                        match_complex_pattern(ep.pattern, pattern,
                                              count=True) /
                        match_complex_pattern(ep.pattern, ep.pattern,
                                              count=True)
                        for pattern in educts
                    ),
                    'energy': ep.energy
                }
                for name, ep in self.energypatterns.items()
            ]
            energies = set(
                diff['name'] for diff in energy_diffs
                if diff['count'] != 0.0
            )
            deltadeltaG = sum(
                diff['energy'] * int(diff['count']) for diff in energy_diffs
                if diff['count'] != 0.0
            )
            if not self.reverse[0]:
                rate = sp.exp(-self.Ea0 - self.phi * deltadeltaG)
            else:
                rate = sp.exp(-self.Ea0 - (1 - self.phi) * deltadeltaG)

            subs = []
            for a in rate.atoms():
                if isinstance(a, Expression):
                    subs.append((a, a.expand_expr(
                        expand_observables=True)))
                elif isinstance(a, Observable):
                    subs.append((a, a.expand_obs()))
            rate = rate.subs(subs)
            rate = sp.powdenest(sp.logcombine(rate, force=True),
                                force=True)
        else:
            energies = set()
            subs = [
                (expr, sp.Symbol(expr.name))
                for expr in model.expressions
            ] + [
                (par, sp.Symbol(par.name))
                for par in model.parameters
            ]
            rate = self.rate.subs(subs)
        return rate, energies

    def compute_educt_mapping(self, educts):
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
            for cp in educts
        ]

        matches = [
            GraphMatcher(
                cp._as_graph(mp_alignment_cp[icp]),
                rp._as_graph(self.graph_diff.mp_alignment_rp[icp],
                             prefix='rp'),
                node_match=node_matcher
            )
            for icp, (rp, cp)
            in enumerate(zip(self.reactant_pattern.complex_patterns,
                             educts))
        ]

        for rpmatch in matches:
            assert (rpmatch.subgraph_is_isomorphic())

        # invert and merge mapping
        return dict(ChainMap(*[
            dict(zip(match.mapping.values(), match.mapping.keys()))
            for match in matches
        ])), mp_alignment_cp


class GraphDiff:
    def __init__(self, rg):
        self.mp_alignment_rp, self.mp_alignment_pp = align_monomer_indices(
            rg.reactant_pattern, rg.product_pattern
        )
        rp_graph = rg.reactant_pattern._as_graph(prefix='rp')
        pp_graph = rg.product_pattern._as_graph(
            prefix='rp', mp_alignment=self.mp_alignment_pp
        )

        self.removed_nodes = tuple(
            n for n, d in rp_graph.nodes(data=True)
            if n not in pp_graph
               or n in pp_graph and pp_graph.nodes[n]['id'] != d['id']
        )
        rp_graph.remove_nodes_from(self.removed_nodes)
        self.added_nodes = tuple(
            (n, d) for n, d in pp_graph.nodes(data=True)
            if n not in rp_graph
        )
        rp_graph.add_nodes_from(self.added_nodes)
        self.removed_edges = tuple(
            nx.difference(rp_graph, pp_graph).edges()
        )
        self.added_edges = tuple(
            nx.difference(pp_graph, rp_graph).edges()
        )
        self.mapped_removed_edges = ()
        self.mapped_added_edges = ()
        self.mapped_removed_nodes = ()
        self.mapped_added_nodes = ()
        self.has_mapping = False

    def apply_mapping(self, mapping):
        for attr in ['removed_edges', 'added_edges']:
            self.__setattr__(
                f'mapped_{attr}',
                tuple(
                    (mapping.get(e[0], e[0]), mapping.get(e[1], e[1]))
                    for e in self.__getattribute__(attr)
                )
            )

        self.mapped_removed_nodes = tuple(
            mapping[n]
            for n in self.removed_nodes
        )
        self.mapped_added_nodes = tuple(
            (mapping.get(n, n), d)
            for n, d in self.added_nodes
        )
        self.has_mapping = True

    def apply(self, ingraph, delete_molecules):
        assert self.has_mapping
        outgraph = ingraph.copy()
        dangling_bonds = []
        if delete_molecules:
            for node in self.mapped_removed_nodes:
                if isinstance(outgraph.nodes[node]['id'], Monomer):
                    neighborhood = nx.ego_graph(outgraph, node, 2)
                    mono_prefix = node.split('_')[0]
                    for n in neighborhood.nodes:
                        if n in self.mapped_removed_nodes:
                            continue  # skip removal here
                        if n.split('_')[0] == mono_prefix:
                            outgraph.remove_node(n)  # remove nodes from
                            # same monomer
                        else:
                            # dont fix dangling bonds here as we might mess
                            # this up again when adding/removing nodes in
                            # the next steps
                            dangling_bonds.append(n)
        outgraph.remove_nodes_from(self.mapped_removed_nodes)
        outgraph.add_nodes_from(self.mapped_added_nodes)
        outgraph.add_edges_from(self.mapped_added_edges)
        outgraph.remove_edges_from(self.mapped_removed_edges)
        # fix dangling bonds:
        if delete_molecules:
            for node in list(dangling_bonds):
                mono_prefix = node.split('_')[0]
                if f'{mono_prefix}_unbound' not in outgraph.nodes():
                    outgraph.add_node(f'{mono_prefix}_unbound', id=NO_BOND)
                outgraph.add_edge(node, f'{mono_prefix}_unbound')
        return outgraph


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
        index = next((
            (icp, imp) for (icp, imp), pp_mono in pp_monos.items()
            if pp_mono == rp_mono
        ), None)
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
    compartments = {n for n, d in graph.nodes(data=True)
                    if isinstance(d['id'], Compartment)}
    components = nx.connected_components(
        graph.subgraph([n for n in graph.nodes if n not in compartments])
    )
    return ReactionPattern([
        complex_pattern_from_graph(graph.subgraph(c | compartments))
        for c in components
    ])


def complex_pattern_from_graph(graph):
    bounds = list()
    compartment = None
    mps = []
    for n, d in graph.nodes(data=True):
        if isinstance(d['id'], Compartment):
            continue
        if isinstance(d['id'], Monomer):
            mps.append(monomer_pattern_from_node(graph, n, bounds))
    return ComplexPattern(mps, compartment
                          if all(mp.compartment is None for mp in mps)
                          else None)


def monomer_pattern_from_node(graph, monomer_node, bounds):
    monomer = graph.nodes[monomer_node]['id']
    compartment = None
    site_conditions = dict()
    for site in graph.neighbors(monomer_node):
        if isinstance(graph.nodes[site]['id'], Compartment):
            compartment = graph.nodes[site]['id']
        else:
            site_conditions[graph.nodes[site]['id']] = \
                site_condition_from_node(graph, monomer, site, bounds)
    return MonomerPattern(monomer, site_conditions, compartment)


def site_condition_from_node(graph, monomer, site_node, bounds):
    states = []
    site = graph.nodes.data()[site_node]['id']
    for condition_node in graph.neighbors(site_node):
        state_candidate = graph.nodes.data()[condition_node]['id']
        if state_candidate == 'NoBond':
            continue
        elif isinstance(state_candidate, Monomer):
            continue
        elif site in monomer.site_states and state_candidate in \
                monomer.site_states[site]:
            states.append(state_candidate)
        else:
            if site_node in bounds:
                states.append(bounds.index(site_node))
            else:
                states.append(len(bounds))
                bounds.append(condition_node)
    if len(states) == 0:
        return None
    elif len(states) == 1:
        return states[0]
    else:
        return tuple(states)
