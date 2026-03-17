# Analyse V7 - Composition et qualite

## Donnees analysees
- Nombre total d'instances: 9000
- Familles: er, mesh, sp

## Statistiques par famille
```
        n_instances  j_mean  j_std  B_mean  ratio_mean  nodes_mean  edges_mean
family                                                                        
er             3000  0.2171 0.1701  8.7362      0.3615      6.4137     15.0057
mesh           3000  0.2364 0.1241 19.9570      0.5986      8.0767     20.6640
sp             3000  0.2765 0.2664  8.7779      0.3146      7.0720      8.4817
```

## Correlations globales (Spearman)
```
                    J_star       B  budget_ratio_total  budget_ratio_path  n_nodes  n_edges  density  n_repairable
J_star              1.0000 -0.0895             -0.1737            -0.8032   0.0266  -0.0859  -0.1778        0.0266
B                  -0.0895  1.0000              0.6851             0.4695   0.7144   0.7448  -0.1510        0.7144
budget_ratio_total -0.1737  0.6851              1.0000             0.4184   0.1294   0.3372   0.2637        0.1294
budget_ratio_path  -0.8032  0.4695              0.4184             1.0000   0.3193   0.3998   0.0165        0.3193
n_nodes             0.0266  0.7144              0.1294             0.3193   1.0000   0.8231  -0.4853        1.0000
n_edges            -0.0859  0.7448              0.3372             0.3998   0.8231   1.0000   0.0493        0.8231
density            -0.1778 -0.1510              0.2637             0.0165  -0.4853   0.0493   1.0000       -0.4853
n_repairable        0.0266  0.7144              0.1294             0.3193   1.0000   0.8231  -0.4853        1.0000
```

## Correlations budget -> J* par famille (Spearman)
- er: corr(J*, B)=-0.5039, corr(J*, B/C_total)=-0.2062, corr(J*, B/C_min_path)=-0.8255
- mesh: corr(J*, B)=0.0171, corr(J*, B/C_total)=-0.6290, corr(J*, B/C_min_path)=-0.6474
- sp: corr(J*, B)=-0.0495, corr(J*, B/C_total)=-0.1548, corr(J*, B/C_min_path)=-0.9234

## Influence budget vs topologie (controle de difficulte)
- Modele R2 famille seule: 0.0157
- Modele R2 famille + difficulte: 0.7525
- Modele R2 famille + difficulte + budget_ratio_total: 0.8476
- Gain R2 du budget (au-dela topologie+difficulte): 0.0951
- Gain R2 attribuable a la topologie (au-dela difficulte+budget): 0.0113
- Coefficient standardise budget_ratio_total (modele complet): -0.0805

## Lecture rapide
- Si le coefficient budget est negatif, augmenter le budget fait baisser J* (a difficulte/topologie comparables).
- Le gain R2 du budget quantifie l'impact marginal du budget independamment de la difficulte et de la famille.
- Le gain R2 de la topologie quantifie l'effet structurel propre a la famille de graphe.