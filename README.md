# proteus
modular, scalable, and flexible network for image understanding

# Code Structure:
This code base would be utilising plugin architecture to make the models dynamic and modular. In its barebone, the codebase would consist of 3 parts:
- Dataloaders: These make the data accessible for train/test.
- Backbone: This is the core feature extractor which generates image embeddings
- Heads: This is task specific. The task of a "head" would be user defined.

# Setup:
After cloning the repo, please run the commands to set up the environment:


Step[1]: Build and activate the virtual env and install pre-commit hooks:

```
sh build_venv.sh
source venv/bin/activate
pre-commit install
```
