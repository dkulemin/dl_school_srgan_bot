## SuperResolution telegram bot

Это итоговый проект первого семестра продвинутого потока курса "Deep Learning" Школы глубокого обучения ФПМИ МФТИ.

Данный бот может принимать картинку маленького разрешения и возвращать эту же картинку, но улучшенного качетсва.

Для запуска бота в командной строке выполните:
```
PYTHONPATH=~/{path-to-the-package}/dl_school_srgan_bot python3 tg_bot/bot.py
```

В проекте используется библиотека [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

Архитектура модели и веса взяты из стороннего проекта [SRGAN-PyTorch](https://github.com/Lornatang/SRGAN-PyTorch)

![screenshot](https://github.com/dkulemin/dl_school_srgan_bot/raw/master/source/screenshots/demo-screenshot.jpg)