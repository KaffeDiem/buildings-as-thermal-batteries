# Buildings as Thermal Batteries

Code and notebooks for a Master Thesis on using Danish electrically heated households as thermal batteries for grid flexibility and cost and emission optimisations.

## Setup 

This project uses [UV](https://github.com/astral-sh/uv) to manage Python environments.

The experiment and the notebooks require two different environments. Common for both is that they can be installed by navigating to the corresponding directory and running:

```bash 
uv sync
```

## Field experiement

> [!NOTE]  
> The field experiment assumes an envioronment file with a smart plug IP and API-keys for the services. These ara not provided.

Code for the dynamic programming field experiemnt is available in the `/experiment` folder with separate results and controller code.


## Notebooks

The `/notebooks` folder provides notebooks and raw data for calculations used in the report.



