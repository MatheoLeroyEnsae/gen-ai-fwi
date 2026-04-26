# gen-ai-fwi

Projet de Generative AI pour l'assurance.
Indicateur climatique considéré : FWI (Fire Weather Index).
Objectif : générer des images similaires aux inputs avec des modèles de type VAE et GAN.

## Données

- **Source** : Mesures historiques quotidiennes du FWI pour l'Espagne (données importées depuis s3)
- **Images** : Images 28x28 construites à partir des valeurs de FWI par grille géographique

## Structure

```
gen-ai-fwi/
├── notebooks/               # Jupyter notebooks
├── models/                # Modèles GAN/VAE
│   ├── DCGAN.py          # DCGAN original
├── callbacks/             # Callbacks TensorBoard/visualisation
└── run/                  # Images générées
└── src/
    ├── models/validation.ipynb.py # CNN to predict seasonality
    ├── stats/ # modules for plotting descriptive statistics 
```

## Notebooks principaux

| Notebook | Description |
|----------|------------|
| `00.data_loading` | Chargement des données .nc |
| `01.0.data_spain_france` | Préparation données spatiales Espagne - France |
| `01.1.data_images` | Création des images 28x28 |
| `01.2.images_loading_28_28` | Chargement images |
| `01.2.images_loading_28_28_modified` | DCGAN sur images normalisées |
| `01.2.images_loading_28_28_normalized_light` | DCGAN léger (latent_dim=64) |
| `03_pca_analysis` | Analyse ACP |
| `04_VAE` | Variational Autoencoder |
| `04-WGANGP-FWI` | WGAN-GP |

## Exécution

```bash
# Environment Onyxia
cd /home/onyxia/gen-ai-fwi
jupyter notebook notebooks/01.2.images_loading_28_28_normalized_light.ipynb
```

### Installation des packages

```bash
pip install -r requirements.txt
```

## Dépendances

- TensorFlow, Keras
- NumPy, Pandas, geopandas
- Matplotlib
- scikit-image, scikit-image
- fidle
- s3fs
- scipy
