from io import BytesIO
from PIL import Image
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters
from textwrap import dedent
import cv2
import numpy as np
import asyncio
from typing import Optional

from model import SuperResolution
from utils import TG_TOKEN_PATH, IMAGE_THRESHOLD


async def help_command(update: Application, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        dedent("""\
            Привет!
            Я бот, который может увеличить разрешение картинки в 4 раза.
            Просто отправь мне картинку без сжатия, только не большую, а то я могу надолго задуматься ;) 
        """)
    )


def _process_photo(photo: bytearray) -> Optional[BytesIO]:
    numpy_image = cv2.imdecode(np.frombuffer(BytesIO(photo).getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if np.sum(numpy_image.shape[:-1]) < IMAGE_THRESHOLD:
        sr = SuperResolution()
        numpy_image = sr(numpy_image)

        pil_image = Image.fromarray(numpy_image)

        bio = BytesIO()
        bio.name = 'image.png'
        pil_image.save(bio, 'png')
        bio.seek(0)

        return bio
    return None


async def echo_image(update: Application, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await context.bot.get_file(update.message.document.file_id)
    b_photo = await photo_file.download_as_bytearray()
    process_photo_result = await asyncio.to_thread(_process_photo, b_photo)
    if process_photo_result:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=await asyncio.to_thread(_process_photo, b_photo)
        )
    else:
        await update.message.reply_text(
            dedent("""\
                Боюсь картинка слишком большая, мне с ней не справиться! :'(
            """)
        )


def main() -> None:
    application = Application.builder().token(TG_TOKEN_PATH.read_text()).build()

    application.add_handler(CommandHandler("start", help_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.IMAGE, echo_image))
    application.run_polling()


if __name__ == "__main__":
    main()
