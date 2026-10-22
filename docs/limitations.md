# Limites

- Pas de carte FPGA physique : pas de mesure de fréquence, puissance ou ressources réelles.
- Localisation de pupille sur images fixes, pas d'estimation du point regardé.
- ROI fixe (400x400), 24 annotations écartées (pupilles hors ROI).
- Résolution d'entrée 16x16 (un pixel = 25 px de la ROI).
- Validation tirée du même participant que l'entraînement (p1) ; test disjoint (p2).
- Quantification INT8/INT32 : écart moyen de 3,2 px entre les prédictions
  flottantes et entières, soit +1,1 px sur l'erreur de localisation
  (4,06 -> 5,17 px). Cause : accumulation d'erreurs sur 3 couches avec
  seulement 127 niveaux d'activation. Correctif possible : activations INT16
  (non implémenté, voir Travaux futurs).
- Identité bit-à-bit VHDL vérifiée sur des vecteurs de test, pas prouvée
  formellement.
