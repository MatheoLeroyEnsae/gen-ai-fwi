# gen-ai-fwi

Projet de Generative AI pour l'assurance incendie (FWI - Fire Weather Index).

## Données

- **Source** : `data/fwi_daily_projection.csv` - projections quotidiennes du FWI pour l'Espagne et le Portugal
- **Images** : Images 28x28 construites à partir des valeurs de FWI par grille géographique

## Structure

```
gen-ai-fwi/
├── data/                    # Données (CSV, numpy)
├── notebooks/               # Jupyter notebooks
├── models/                # Modèles GAN/VAE
│   ├── DCGAN.py          # DCGAN original
│   ├── DCGAN_tuned.py   # DCGAN corrigé
│   └── CGAN.py          # CGAN conditionnel (par mois)
├── callbacks/             # Callbacks TensorBoard/visualisation
└── run/                  # Images générées
```

## Notebooks principaux

| Notebook | Description |
|----------|------------|
| `00.data_loading` | Chargement des données CSV |
| `01.0.data_spain_france` | Préparation données spatiales |
| `01.1.data_images` | Création des images 28x28 |
| `01.2.images_loading_28_28` | Chargement images |
| `01.2.images_loading_28_28_modified` | DCGAN sur images normalisées |
| `01.2.images_loading_28_28_normalized_light` | DCGAN léger (latent_dim=64) |
| `01.2.images_loading_28_28_cgan` | CGAN conditionnel par mois |
| `03_VAE` | Variational Autoencoder |
| `04-WGANGP-FWI` | WGAN-GP |

## Exécution

```bash
# Environment Onyxia
cd /home/onyxia/gen-ai-fwi
jupyter notebook notebooks/01.2.images_loading_28_28_normalized_light.ipynb
```

## Modèles

- **DCGAN** : Deep Convolutional GAN sur images 28x28
- **DCGAN_tuned** : Version corrigée pour `predict()`
- **CGAN** : Conditional GAN avec le mois en entrée

## Dépendances

- TensorFlow, Keras
- NumPy, Pandas
- Matplotlib
- scikit-image