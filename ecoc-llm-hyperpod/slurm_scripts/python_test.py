import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout
)


logger = logging.getLogger(__name__)

logger.info("Hellooo this is from the logger file 2 .....")