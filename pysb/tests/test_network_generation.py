from pysb.network_expansion import NetworkExpansion
from pysb.bng import generate_equations
from pysb import (
    Model, Monomer, Initial, Rule, EnergyPattern, Expression, Parameter,
    ComplexPattern
)
from .test_importers import _bngl_location, model_from_bngl
from pysb.pattern import match_complex_pattern

import sympy as sp

import re


def bngl_compare_network_generation(bng_file):
    m = model_from_bngl(bng_file)

    generate_equations(m, verbose=True)

    network = NetworkExpansion(m)
    network.generate([
        init.pattern for init in m.initials
    ])

    #validate_network(m, network)


def test_bng_models():
    for filename in (#'CaOscillate_Func',
                     #'continue',
                     #'deleteMolecules',
                     'egfr_net',
                     #'empty_compartments_block',
                     #'gene_expr',
                     #'gene_expr_func',
                     #'gene_expr_simple',
                     #'isomerization',
                     #'michment',
                     ##'Motivating_example_cBNGL',
                     #'motor',
                     #'simple_system',
                     #'test_compartment_XML',
                     #'test_setconc',
                     #'test_synthesis_cBNGL_simple',
                     ##'test_synthesis_complex',
                     ##'test_synthesis_complex_0_cBNGL',
                     ##'test_synthesis_complex_source_cBNGL',
                     #'test_synthesis_simple',
                     ##'toy-jim',
                     ##'univ_synth',
                     #'visualize',
                     #'statfactor',
                     ):
        full_filename = _bngl_location(filename)
        yield (bngl_compare_network_generation, full_filename)


def test_enzymatic_catalysis():
    model = Model()
    S = Monomer('S', sites=['e', 'mod'], site_states={'mod': ['u', 'p']})
    K = Monomer('K', sites=['s'])
    P = Monomer('P', sites=['s'])
    ATP = Monomer('ATP')
    ADP = Monomer('ADP')

    for partners in [('K', 'S', 's', 'e'), ('P', 'S', 's', 'e')]:
        prefix = f'bind_{partners[0]}{partners[1]}'
        kon = Parameter(f'{prefix}_kon', 1.0)
        kD = Parameter(f'{prefix}_kD', 1.0)
        phi = Parameter(f'{prefix}_phi', 0.5)
        Ea0 = Expression(f'{prefix}_Ea0', -(sp.ln(kon) + phi * sp.ln(kD)))

        Rule(prefix,
             model.monomers[partners[0]](**{partners[2]: None})
             +
             model.monomers[partners[1]](**{partners[3]: None})
             |
             model.monomers[partners[0]](**{partners[2]: 1})
             %
             model.monomers[partners[1]](**{partners[3]: 1}),
             phi, Ea0, energy=True)

        EnergyPattern(f'{prefix}_ep',
                      model.monomers[partners[0]](**{partners[2]: 1})
                      %
                      model.monomers[partners[1]](**{partners[3]: 1}),
                      Expression(f'{prefix}_deltaG', sp.ln(kD)))

    prefix = 'phosphorylation_S'
    Rule(prefix,
         S(e=1, mod='u') % K(s=1) + ATP() | S(e=1, mod='p') % K(s=1) + ADP(),
         Parameter(f'{prefix}_phi', 0.5),
         Expression(f'{prefix}_deltaG',
                    sp.log(Parameter(f'{prefix}_expdeltaG', 0.0))),
         energy=True)
    prefix = 'dephosphorylation_S'
    Rule(prefix,
         S(e=1, mod='p') % P(s=1) + ATP() | S(e=1, mod='u') % P(s=1) + ADP(),
         Parameter(f'{prefix}_phi', 0.5),
         Expression(f'{prefix}_deltaG',
                    sp.log(Parameter(f'{prefix}_expdeltaG', 0.0))),
         energy=True)

    E_ATP = Expression('E_E0_ATP', sp.log(Parameter('expG0_ATP', 1.0)))
    EnergyPattern('EP_E0_ATP', ComplexPattern([ATP()], None), E_ATP)
    EnergyPattern('EP_pS', ComplexPattern([S(mod='p')], None), E_ATP)

    EnergyPattern('EP_pSK', S(e=1, mod='p') % K(s=1),
                  Expression('E_pSK', sp.log(Parameter('pSK_expdeltaG', 1.0))))
    EnergyPattern('EP_uSP', S(e=1, mod='u') % P(s=1),
                  Expression('E_uSP', sp.log(Parameter('uSP_expdeltaG', 1.0))))

    Initial(ATP(), Parameter('ATP_0', 10), fixed=True)
    Initial(ADP(), Parameter('ADP_0', 0.1), fixed=True)
    Initial(K(s=None), Parameter('K_0', 1.0))
    Initial(P(s=None), Parameter('P_0', 1.0))
    Initial(S(e=None, mod='u'), Parameter('S_0', 1.0))

    generate_equations(model, verbose=True)

    network = NetworkExpansion(model)
    network.generate([
        ep.pattern for ep in model.energypatterns
    ])

    validate_network(model, network)


def validate_network(model, network):
    species_mapper = {
        ispecies_network: next((
            ispecies_model
            for ispecies_model, species_model in enumerate(model.species)
            if match_complex_pattern(species_network, species_model,
                                     exact=True)
        ), None)
        for ispecies_network, species_network in enumerate(network.species)
    }
    for mapping in species_mapper.values():
        assert mapping is not None
    assert len(network.species) == len(model.species)
    subs_rates = {
        sp.Symbol(rate.name): sp.powdenest(sp.logcombine(
            rate.expand_expr(expand_observables=True), force=True
        ), force=True)
        for rate in model.expressions
        if re.search(r'\_local[0-9]+$', rate.name)
    }
    subs_states_temp = {
        sp.Symbol(f'__s{ix}'): sp.Symbol(f'__x{ix}')
        for ix in species_mapper
    }
    subs_states = {
        sp.Symbol(f'__x{ix}'): sp.Symbol(f'__s{species_mapper[ix]}')
        for ix in species_mapper
    }
    reaction_mapper = {
        ireaction_network: next((
            ireaction_model
            for ireaction_model, reaction_model in enumerate(model.reactions)
            if reaction_network['rule'].replace('__reverse', '') ==
               reaction_model['rule'][0]
            and reaction_network['rule'].endswith('__reverse') ==
               reaction_model['reverse'][0]
            and sorted([educt for educt in reaction_model['reactants']]) ==
                sorted([species_mapper[educt] for educt in reaction_network['educts']])
            and sorted([product for product in reaction_model['products']]) ==
                sorted([species_mapper[educt] for educt in reaction_network[
                    'products']])
            and sp.simplify(
                reaction_network['rate'].subs(
                    subs_states_temp
                ).subs(subs_states) - reaction_model['rate'].subs(
                    subs_rates
                )
            ).is_zero
        ), None)
        for ireaction_network, reaction_network in enumerate(network.reactions)
    }
    for mapping in reaction_mapper.values():
        assert mapping is not None
    assert len(network.reactions) == len(model.reactions)
