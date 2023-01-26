## SuperResolution telegram bot

### Описание

Название: SuperResolution Bot (@dkulemin_bot)

Это итоговый проект первого семестра продвинутого потока курса "Deep Learning" Школы глубокого обучения ФПМИ МФТИ.

Данный бот может принимать картинку маленького разрешения и возвращать эту же картинку, но улучшенного качетсва.

В проекте используется библиотека [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

Архитектура модели и веса взяты из стороннего проекта [SRGAN-PyTorch](https://github.com/Lornatang/SRGAN-PyTorch)

### Установка

Команда для клонирования репозитория:
```
git clone https://github.com/dkulemin/dl_school_srgan_bot.git
```

Команда для запуска бота:
```
PYTHONPATH=~/{path-to-the-package}/dl_school_srgan_bot python3 tg_bot/bot.py
```

### Demo

Бот развернут в сервисе [PythonAnywhere](https://www.pythonanywhere.com/), если не доступен можно написать мне в телеграмм: @dkulemin

Видео-демонстрация на [YouTube](https://youtu.be/B8EqLn6NpgQ)

![screenshot](https://github.com/dkulemin/dl_school_srgan_bot/raw/master/source/screenshots/demo-screenshot.jpg)
