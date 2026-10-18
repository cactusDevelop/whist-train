# whist-train

Entrainement headless (CPU) des IA du jeu de whist : moteur Python, agents
PPO, entrainement des jokers. Code seulement : les checkpoints circulent
chiffres (release `checkpoint-latest`).

```
pip install -r requirements.txt
python scripts/train.py --no-godot-sync --no-git-push --no-record-samples --until AAAA-MM-JJTHH:MM
python scripts/train_jokers.py --resume --workers 4 --until AAAA-MM-JJTHH:MM
```
