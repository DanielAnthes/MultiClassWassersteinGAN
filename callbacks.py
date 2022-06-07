import tensorflow as tf


class PrettyMetricPrinter(tf.keras.callbacks.Callback):

    def __init__(self, metrics_to_print, val_metrics_to_print):
        self.metrics_to_print = metrics_to_print
        self.val_metrics_to_print = val_metrics_to_print

    def on_epoch_end(self, epoch, logs):
        printstring = ""
        for m in logs:
            if m in self.metrics_to_print:
                printstring += f"{m}: {logs[m]}\n"
        print(printstring)

    def on_test_end(self, logs):
        printstring= ""
        for m in logs:
            if m in self.val_metrics_to_print:
                printstring += f"val_{m}: {logs[m]}\n"
        print(printstring)

