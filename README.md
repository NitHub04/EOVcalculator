# EOVcalculator
Exercise-induced Oscillatory Ventillation (EOV) calculator: Determine if a HF patient has Exercise Oscillatory Ventilation based on rules outlined in https://doi.org/10.1378/chest.07-2146 Use the data export file from Cosmed CPEX machines for existing code

Why EOV?
Periodic breathing pattern is seen in advanced heart failure.
(Possible pathophysiology) Increased chemosensitivity triggers a cycle of hyperventilation-induced reduction in Paco2 until the apnea threshold is approached and then hypoventilation until Paco2 rises and hyperventilation resumes. 
(EOV= manifestation of PB) Instability of ventilatory control is frequent in patients with heart failure (HF) and may manifest as exercise oscillatory ventilation (EOV).
This can be seen in CPEX as a feature with poor prognosis.

Definition of EOV:
Plot Ve (Minute ventilation vs time)
A ≥ 25% variation of the amplitude of minute ventilation (Ve) persisting for ≥ 60% of exercise duration (Warm-up and exercise phases).
The amplitude of oscillatory ventilation was defined as (peak Ve − mean of (nadir1 Ve and nadir 2 Ve)/(nadir1 Ve and nadir 2 Ve))

Rationale:
Whereas visual estimates have been used previously, the data can now be quanittatively analysed to measure variation and diagnose pathological breathing patterns and EOV.

Specifics:
This script is designed data import from Cosmed UK v2 'breath-by-breath' CPEX data reports.
However it can be modified to use input Minute ventilation, time, phase data from any system.

