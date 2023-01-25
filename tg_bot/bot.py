from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters
from textwrap import dedent
import cv2
import numpy as np

from model import SuperResolution
from utils import TG_TOKEN_PATH


async def help_command(update: Application, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
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


async def echo_image(update: Application, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await context.bot.get_file(update.message.document.file_id)

    b_photo = BytesIO(await photo_file.download_as_bytearray())

    await context.bot.send_document(chat_id=update.effective_chat.id, document=_process_photo(b_photo))


def main() -> None:
    application = Application.builder().token(TG_TOKEN_PATH.read_text()).build()

    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.IMAGE, echo_image))

    application.run_polling()


if __name__ == "__main__":
    main()
