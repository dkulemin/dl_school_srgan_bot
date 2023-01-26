from pathlib import Path


REPO_PATH = Path(__file__).parent.parent
TG_TOKEN_PATH = REPO_PATH / 'source' / 'tokens' / 'tg_bot_token.txt'
GENERATOR_PATH = REPO_PATH / 'source' / 'weights' / 'SRResNet_x4-ImageNet-6dd5216c.pth.tar'

IMAGE_THRESHOLD = 601