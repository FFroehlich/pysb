from pysb.network_expansion import partial_network_expansion
from pysb.bng import generate_equations
from pysb import (
    Model, Monomer, Initial, Rule, EnergyPattern, Expression, Parameter,
    ComplexPattern
)

import sympy as sp

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

    species, reactions = partial_network_expansion(model, [
        ep.pattern for ep in model.energypatterns
    ])

    assert len(species) == len(model.species)
    assert len(reactions) == len(model.reactions)


