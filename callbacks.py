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


class CustomTensorBoard(tf.keras.callbacks.Callback):

    def __init__(self, logdir, log_batches=True):
        # set up directories
        if logdir[-1] == '/':
            logdir = logdir[:-1]
        self.logdir_train = logdir + "_train"
        self.logdir_val = logdir + "_val"
        self.train_writer = tf.summary.create_file_writer(self.logdir_train)
        self.val_writer = tf.summary.create_file_writer(self.logdir_val)
        self.log_batches = log_batches
        self.global_batch = 0

    def on_epoch_begin(self,epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        with self.train_writer.as_default():
            for log in logs:
                tag = 'epoch_' + log
                value = logs[log]
                tf.summary.scalar(tag, value, step=epoch)

    def on_train_batch_begin(self, epoch, logs):
        pass

    def on_train_batch_end(self, epoch, logs):
        if self.log_batches:
            with self.train_writer.as_default():
                for log in logs:
                    tag = 'batch_' + log
                    value = logs[log]
                    tf.summary.scalar(tag, value, step=self.global_batch)
                    self.global_batch += 1

    def on_test_begin(self, logs):
        pass

    def on_test_end(self, logs):
        with self.val_writer.as_default():
            for log in logs:
                tag = 'epoch_' + log
                value = logs[log]
                tf.summary.scalar(tag, value, step=self.epoch)

