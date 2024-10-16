import argparse
from datetime import datetime, timedelta

# Clase personalizada para el manejo del argumento verbose
class VAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(VAction, self).__init__(option_strings, dest, nargs, const,
                                      default, type, choices, required,
                                      help, metavar)
        self.values = 0

    def __call__(self, parser, args, values, option_string=None):
        if values is None:
            self.values += 1
        else:
            try:
                self.values = int(values)
            except ValueError:
                self.values = values.count('v') + 1
        setattr(args, self.dest, self.values)

# Función para obtener las fechas por defecto
def get_default_dates():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    default_from = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    default_until = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
    return default_from.strftime('%Y-%m-%d %H:%M:%S'), default_until.strftime('%Y-%m-%d %H:%M:%S')

# Clase principal que manejará los argumentos
class BatchProcess:
    def __init__(self):
        self.config = self.parse_args()

    def parse_args(self):
        # Obtener las fechas por defecto
        default_from, default_until = get_default_dates()

        # Crear el parser de argumentos
        parser = argparse.ArgumentParser(description='Ejecución de procesos en batch.')
        
        # Agregar los argumentos
        parser.add_argument('-f', '--from_time', type=str, default=default_from, help='Fecha de inicio de búsqueda (formato: YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('-u', '--until', type=str, default=default_until, help='Fecha de finalización de búsqueda (formato: YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('-v', '--verbose', nargs='?', action=VAction, dest='verbose', help='Nivel de verbosidad', default=0)

        # Parsear los argumentos
        args = parser.parse_args()
        return args

    # Función que ejecutará el proceso en batch
    def run(self):
        from_time = self.config.from_time  # Actualizamos el nombre aquí
        until_time = self.config.until
        verbose = self.config.verbose

        # Lógica de verbosidad
        if verbose >= 1:
            print(f"Parámetros recibidos: --from_time {from_time}, --until {until_time}, --verbose {verbose}")

        # Aquí se incluirían las operaciones principales del proceso
        self.main_process()

    # Función que simula el proceso principal
    def main_process(self):
        verbose = self.config.verbose

        # Paso principal
        if verbose >= 1:
            print("Ejecutando el proceso principal...")

        # Ejemplo de bucle (simulación)
        for i in range(3):
            if verbose >= 2:
                print(f"Iteración {i} en el bucle principal.")

            # Simulación de trabajo
            if verbose >= 3:
                print(f"Objeto importante en la iteración {i}: {{'clave': 'valor'}}")

        if verbose >= 1:
            print("Proceso principal finalizado.")

# Ejecutar el proceso
if __name__ == "__main__":
    batch_process = BatchProcess()
    batch_process.run()
