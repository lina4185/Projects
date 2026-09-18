import time

class RealTimeDataIntegrator:

    def adaptive_polling(self, severe_weather=False):

        if severe_weather:
            interval = 60

        else:
            interval = 300

        return interval

    def run(self):

        while True:

            print("Fetching real-time updates...")

            time.sleep(self.adaptive_polling())