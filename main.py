"""
Authors: Krzysztof Skwira & Tomasz Lemke
To run program install
pip install scikit-fuzzy
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
The Tipping Problem
-------------------

Let's create a fuzzy control system which models how you might choose to tip
at a restaurant.  When tipping, you consider the service and food quality,
rated between 0 and 10.  You use this to leave a tip of between 0 and 25%.

We would formulate this problem as:

* Antecednets (Inputs)
   - `service`
      * Universe (ie, crisp value range): How good was the service of the wait
        staff, on a scale of 0 to 10?
      * Fuzzy set (ie, fuzzy value range): poor, acceptable, amazing
   - `food quality`
      * Universe: How tasty was the food, on a scale of 0 to 10?
      * Fuzzy set: bad, decent, great
* Consequents (Outputs)
   - `tip`
      * Universe: How much should we tip, on a scale of 0% to 25%
      * Fuzzy set: low, medium, high
* Rules
   - IF the *service* was good  *or* the *food quality* was good,
     THEN the tip will be high.
   - IF the *service* was average, THEN the tip will be medium.
   - IF the *service* was poor *and* the *food quality* was poor
     THEN the tip will be low.
* Usage
   - If I tell this controller that I rated:
      * the service as 9.8, and
      * the quality as 6.5,
   - it would recommend I leave:
      * a 20.2% tip.


Creating the Tipping Controller Using the skfuzzy control API
-------------------------------------------------------------

We can use the `skfuzzy` control system API to model this.  First, let's
define fuzzy variables
"""
# Input your age to calculate the maximum heart rate
age = 35
voMax = 220 - age

# New Antecedent/Consequent objects hold universe variables and membership functions
time = ctrl.Antecedent(np.arange(0, 151, 1), 'total time (min)')
heartRate = ctrl.Antecedent(np.arange(0.5*voMax, voMax+1, 1), 'heart rate (BPM)')
pace = ctrl.Antecedent(np.arange(6, 20.1, 0.1), 'average pace (km/h)')

trainingEffect = ctrl.Consequent(np.arange(0, 6, 1), 'training effect')

# Auto-membership function population is possible with .automf(3, 5, or 7)
# Custom membership functions can be built interactively with a familiar, Pythonic API

time['short'] = fuzz.trimf(time.universe, [0, 0, 20])
time['medium'] = fuzz.trimf(time.universe, [15, 30, 45])
time['long'] = fuzz.trimf(time.universe, [40, 55, 70])
time['very long'] = fuzz.trimf(time.universe, [60, 90, 120])
time['ultra long'] = fuzz.trimf(time.universe, [100, 150, 150])

heartRate.automf(5, 'quality',
                 ['Moderate Activity', 'Weight Control', 'Aerobic', 'Anaerobic', 'VO2 Max'])

pace['very slow'] = fuzz.trimf(pace.universe, [6, 6, 8])
pace['slow'] = fuzz.trimf(pace.universe, [7.5, 9, 10])
pace['medium'] = fuzz.trimf(pace.universe, [9.5, 11, 12])
pace['fast'] = fuzz.trimf(pace.universe, [11.5, 14, 16.5])
pace['very fast'] = fuzz.trimf(pace.universe, [14.5, 20, 20])


trainingEffect.automf(6, 'quality',
                      ['No effect', 'Minor effect', 'Maintaining', 'Improving', 'Highly improving', 'Overloading!'])


"""
To help understand what the membership looks like, use the ``view`` methods.
"""
# time.view()
# heartRate.view()
# pace.view()

# trainingEffect.view()

"""
Fuzzy rules
-----------

Now, to make these triangles useful, we define the *fuzzy relationship*
between input and output variables. For the purposes of our example, consider
three simple rules:

1. If the food is poor OR the service is poor, then the tip will be low
2. If the service is average, then the tip will be medium
3. If the food is good OR the service is good, then the tip will be high.

Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable tip is a challenge. This is the
kind of task at which fuzzy logic excels.
"""

rule1 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long'] | time['medium']) & heartRate['VO2 Max'] & pace['very fast'], trainingEffect['Overloading!'])
rule2 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['VO2 Max'] & pace['fast'], trainingEffect['Overloading!'])
rule3 = ctrl.Rule((time['ultra long'] | time['very long']) & heartRate['VO2 Max'] & pace['medium'], trainingEffect['Overloading!'])
rule4 = ctrl.Rule(time['ultra long'] & heartRate['Anaerobic'] & (pace['fast'] | pace['very fast']), trainingEffect['Overloading!'])

rule5 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long'] | time['medium']) & heartRate['VO2 Max'] & pace['very slow'], trainingEffect['Highly improving'])
rule6 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['VO2 Max'] & pace['slow'], trainingEffect['Highly improving'])
rule7 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['VO2 Max'] & pace['medium'], trainingEffect['Highly improving'])
rule8 = ctrl.Rule((time['medium'] | time['short']) & heartRate['VO2 Max'] & pace['fast'], trainingEffect['Highly improving'])
rule9 = ctrl.Rule(time['short'] & heartRate['VO2 Max'] & pace['very fast'], trainingEffect['Highly improving'])
rule10 = ctrl.Rule(time['ultra long'] & heartRate['Anaerobic'] & (pace['medium'] | pace['slow']), trainingEffect['Highly improving'])
rule11 = ctrl.Rule(time['very long'] & heartRate['Anaerobic'] & (pace['fast'] | pace['medium']), trainingEffect['Highly improving'])
rule12 = ctrl.Rule(time['long'] & heartRate['Anaerobic'] & pace['fast'], trainingEffect['Highly improving'])
rule13 = ctrl.Rule((time['very long'] | time['long'] | time['medium']) & heartRate['Anaerobic'] & pace['very fast'], trainingEffect['Highly improving'])
rule14 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['Aerobic'] & pace['very fast'], trainingEffect['Highly improving'])
rule15 = ctrl.Rule(time['ultra long'] & heartRate['Aerobic'] & (pace['fast'] | pace['medium']), trainingEffect['Highly improving'])
rule16 = ctrl.Rule(time['very long'] & heartRate['Aerobic'] & pace['fast'], trainingEffect['Highly improving'])

rule17 = ctrl.Rule(time['short'] & heartRate['VO2 Max'] & pace['very slow'], trainingEffect['Improving'])
rule18 = ctrl.Rule((time['medium'] | time['short']) & heartRate['VO2 Max'] & pace['slow'], trainingEffect['Improving'])
rule19 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['Anaerobic'] & pace['very slow'], trainingEffect['Improving'])
rule20 = ctrl.Rule((time['very long'] | time['long'] | time['medium']) & heartRate['Anaerobic'] & pace['slow'], trainingEffect['Improving'])
rule21 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Anaerobic'] & pace['medium'], trainingEffect['Improving'])
rule22 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Anaerobic'] & pace['fast'], trainingEffect['Improving'])
rule23 = ctrl.Rule(time['short'] & heartRate['Anaerobic'] & pace['very fast'], trainingEffect['Improving'])
rule24 = ctrl.Rule((time['ultra long'] | time['very long']) & heartRate['Aerobic'] & pace['very slow'], trainingEffect['Improving'])
rule25 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['Aerobic'] & pace['slow'], trainingEffect['Improving'])
rule26 = ctrl.Rule((time['very long'] | time['long'] | time['medium']) & heartRate['Aerobic'] & pace['medium'], trainingEffect['Improving'])
rule27 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Aerobic'] & pace['fast'], trainingEffect['Improving'])
rule28 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Aerobic'] & pace['very fast'], trainingEffect['Improving'])
rule29 = ctrl.Rule(time['ultra long'] & heartRate['Weight Control'] & pace['medium'], trainingEffect['Improving'])
rule30 = ctrl.Rule((time['ultra long'] | time['very long']) & heartRate['Weight Control'] & pace['fast'], trainingEffect['Improving'])
rule31 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['Weight Control'] & pace['very fast'], trainingEffect['Improving'])
rule32 = ctrl.Rule(time['ultra long'] & heartRate['Moderate Activity'] & pace['fast'], trainingEffect['Improving'])
rule33 = ctrl.Rule((time['ultra long'] | time['very long']) & heartRate['Moderate Activity'] & pace['very fast'], trainingEffect['Improving'])

rule34 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Anaerobic'] & pace['very slow'], trainingEffect['Maintaining'])
rule35 = ctrl.Rule(time['short'] & heartRate['Anaerobic'] & pace['slow'], trainingEffect['Maintaining'])
rule36 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Aerobic'] & pace['very slow'], trainingEffect['Maintaining'])
rule37 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Aerobic'] & pace['slow'], trainingEffect['Maintaining'])
rule38 = ctrl.Rule(time['short'] & heartRate['Aerobic'] & pace['medium'], trainingEffect['Maintaining'])
rule39 = ctrl.Rule((time['ultra long'] | time['very long']) & heartRate['Weight Control'] & pace['very slow'], trainingEffect['Maintaining'])
rule40 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['Weight Control'] & pace['slow'], trainingEffect['Maintaining'])
rule41 = ctrl.Rule((time['very long'] | time['long'] | time['medium']) & heartRate['Weight Control'] & pace['medium'], trainingEffect['Maintaining'])
rule42 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Weight Control'] & pace['fast'], trainingEffect['Maintaining'])
rule43 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Weight Control'] & pace['very fast'], trainingEffect['Maintaining'])
rule44 = ctrl.Rule(time['ultra long'] & heartRate['Moderate Activity'] & pace['slow'], trainingEffect['Maintaining'])
rule45 = ctrl.Rule((time['ultra long'] | time['very long']) & heartRate['Moderate Activity'] & pace['medium'], trainingEffect['Maintaining'])
rule46 = ctrl.Rule((time['very long'] | time['long'] | time['medium']) & heartRate['Moderate Activity'] & pace['fast'], trainingEffect['Maintaining'])
rule47 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Moderate Activity'] & pace['very fast'], trainingEffect['Maintaining'])

rule48 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Weight Control'] & pace['very slow'], trainingEffect['Minor effect'])
rule49 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Weight Control'] & pace['slow'], trainingEffect['Minor effect'])
rule50 = ctrl.Rule(time['short'] & heartRate['Weight Control'] & pace['medium'], trainingEffect['Minor effect'])
rule51 = ctrl.Rule((time['ultra long'] | time['very long'] | time['long']) & heartRate['Moderate Activity'] & pace['very slow'], trainingEffect['Minor effect'])
rule52 = ctrl.Rule((time['very long'] | time['long'] | time['medium']) & heartRate['Moderate Activity'] & pace['slow'], trainingEffect['Minor effect'])
rule53 = ctrl.Rule((time['long'] | time['medium'] | time['short']) & heartRate['Moderate Activity'] & pace['medium'], trainingEffect['Minor effect'])
rule54 = ctrl.Rule(time['short'] & heartRate['Moderate Activity'] & pace['fast'], trainingEffect['Minor effect'])

rule55 = ctrl.Rule((time['medium'] | time['short']) & heartRate['Moderate Activity'] & pace['very slow'], trainingEffect['No effect'])
rule56 = ctrl.Rule(time['short'] & heartRate['Moderate Activity'] & pace['slow'], trainingEffect['No effect'])

rule33.view()

"""
Control System Creation and Simulation
---------------------------------------

Now that we have our rules defined, we can simply create a control system
via:
"""

training_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                    rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
                                    rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28,
                                    rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37,
                                    rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46,
                                    rule47, rule48, rule49, rule50, rule51, rule52, rule53, rule54, rule55,
                                    rule56])

"""
In order to simulate this control system, we will create a
``ControlSystemSimulation``.  Think of this object representing our controller
applied to a specific set of circumstances.  For tipping, this might be tipping
Sharon at the local brew-pub.  We would create another
``ControlSystemSimulation`` when we're trying to apply our ``tipping_ctrl``
for Travis at the cafe because the inputs would be different.
"""

training = ctrl.ControlSystemSimulation(training_ctrl)

"""
We can now simulate our control system by simply specifying the inputs
and calling the ``compute`` method.  Suppose we rated the quality 6.5 out of 10
and the service 9.8 of 10.
"""
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)

training.input['total time (min)'] = 0
training.input['heart rate (BPM)'] = 0
training.input['average pace (km/h)'] = 0


# Crunch the numbers
training.compute()

"""
Once computed, we can view the result as well as visualize it.
"""

# print(training.output['training effect'])

effort = round(training.output['training effect'], 2)
print(f"Your training effort is: {effort}")
trainingEffect.view(sim=training)

if 0 <= effort <= 0.99:
    print("No effect")
elif 1 <= effort <= 1.99:
    print("Minor effect")
elif 2 <= effort <= 2.99:
    print("Maintaining")
elif 3 <= effort <= 3.99:
    print("Improving")
elif 4 <= effort <= 4.99:
    print("Highly improving")
elif 5 <= effort:
    print("Overloading!")
