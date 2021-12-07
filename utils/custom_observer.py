from sacred.observers import RunObserver
import os
import shutil

STARTED_FILE = 'started_event.txt'
COMPLETED_FILE = 'completed_event.txt'

class CustomObserver(RunObserver):
    
    def __init__(self, dest):
        self.dest = dest
    
    def queued_event(self, ex_info, command, queue_time, config, meta_info,
                     _id):
        pass

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):

        # Create complete string
        out = ["Start time:\t{}".format(start_time),
               "\nex_info:\t{}".format(ex_info),
               "\ncommand:\t{}".format(command),
               "\nhost_info:\t{}".format(host_info),
               ]
        out = "\n".join(out)

        # Write fail trace to disk
        fn = os.path.join(self.dest, STARTED_FILE)
        with open(fn,'w') as f:
            f.write(out)


    def heartbeat_event(self, info, captured_out, beat_time, result):
        pass

    def completed_event(self, stop_time, result):
        
        # Create complete string
        out = "Complete time:\t{}\n\n".format(stop_time)

        # Write fail trace to disk
        fn = os.path.join(self.dest, COMPLETED_FILE)
        with open(fn,'w') as f:
            f.write(out)
            
        return True 


    def interrupted_event(self, interrupt_time, status):
        dest_new = self.dest + '_INTERRUPTED'
        if os.path.exists(dest_new):
            shutil.rmtree(dest_new)
        os.rename(self.dest, dest_new)
        
    def failed_event(self, fail_time, fail_trace):

        # Create fail string
        out = "Failed time:\t{}\n\n".format(fail_time) + \
                                                   "\n".join(fail_trace)

        # Write fail trace to disk
        fn = os.path.join(self.dest, 'failed_event.txt')
        with open(fn,'w') as f:
            f.write(out)

        # Rename destination folder
        dest_new = self.dest + '_FAILED'
        if os.path.exists(dest_new):
            shutil.rmtree(dest_new)
        os.rename(self.dest, dest_new)

    def resource_event(self, filename):
        pass

    def artifact_event(self, name, filename):
        pass
        
