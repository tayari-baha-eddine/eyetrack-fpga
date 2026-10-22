# Format fixed-point

| Element | Representation |
| --- | --- |
| Pixel d'entree | uint8 0..255, valeur reelle = p / 255 |
| Poids | int8 dans [-127, 127], un facteur d'echelle par couche (symetrique) |
| Biais | int32 = arrondi(biais_reel / (echelle_entree * echelle_poids)) |
| Accumulateur | int32 signe : somme des produits + biais |
| Couches 1 et 2, sortie | a = saturer( (acc * M + 2^(S-1)) >> S , 0 , 127 ) |
| Couche 3, sortie | accumulateur int32 brut ; valeur reelle = acc * out_scale |
| Pixels de la ROI | valeur reelle * 399 |
