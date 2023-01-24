from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import Updater, CallbackContext, CommandHandler, Filters, MessageHandler, BaseFilter
from textwrap import dedent
import cv2
import numpy as np

from model import SuperResolution
from utils import TG_TOKEN_PATH


def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        dedent("""\
            Бот может увеличить разрешение переданной картинки в 4 раза.
            Все что нужно сделать - отправить картинку без сжатия.
        """)
    )


def _process_photo(photo: BytesIO) -> BytesIO:
    numpy_image = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    sr = SuperResolution()
    numpy_image = sr(numpy_image)

    pil_image = Image.fromarray(numpy_image)

    bio = BytesIO()
    bio.name = 'image.jpeg'
    pil_image.save(bio, 'JPEG')
    bio.seek(0)

    return bio


def echo_image(update: Update, context: CallbackContext) -> None:
    photo_file = context.bot.get_file(update.message.document.file_id)

    b_photo = BytesIO(photo_file.download_as_bytearray())

    context.bot.send_document(chat_id=update.effective_chat.id, document=_process_photo(b_photo))


def main() -> None:
    updater = Updater(TG_TOKEN_PATH.read_text())

    updater.dispatcher.add_handler(CommandHandler("help", help_command))
    updater.dispatcher.add_handler(MessageHandler(Filters.update.message & Filters.document, echo_image))

    updater.start_polling()

    print('Started')

    updater.idle()


if __name__ == "__main__":
    main()
