class RetryManager:
    """
    Handles retry decisions for failed documents.
    """

    MAX_RETRIES = 3

    def __init__(self):
        pass

    def retry(self, document):
        """
        Retry processing if validation failed.
        """

        # Validation successful
        if document.validation_passed:
            document.mark_completed()
            return False

        # Retry available
        if document.retry_count < self.MAX_RETRIES:
            document.retry_count += 1
            document.status = "RETRY"
            return True

        # Retry limit exceeded
        document.mark_failed("Retry limit exceeded")
        return False
