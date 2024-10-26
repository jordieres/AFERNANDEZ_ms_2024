import argparse
from datetime import datetime, timedelta

class VAction(argparse.Action):
    """
    Clase personalizada para manejar el argumento de verbosidad.

    Esta clase permite contar la cantidad de veces que se utiliza el argumento 
    `-v` o `--verbose`, así como recibir un valor numérico opcional.

    Args:
        option_strings (list): Lista de cadenas de opción que corresponden 
            a esta acción.
        dest (str): Nombre del atributo al que se asignará el valor.
        nargs (int, optional): Número de argumentos a consumir. Default es None.
        const (any, optional): Valor constante a utilizar si nargs es `?`.
        default (any, optional): Valor por defecto a usar si no se proporciona.
        type (type, optional): Tipo de argumento que se espera.
        choices (list, optional): Opciones válidas para el argumento.
        required (bool, optional): Si es True, el argumento es obligatorio.
        help (str, optional): Descripción de ayuda para el argumento.
        metavar (str, optional): Nombre a usar en el mensaje de ayuda.

    Attributes:
        values (int): Contador de niveles de verbosidad.
    """

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(VAction, self).__init__(option_strings, dest, nargs, const,
                                      default, type, choices, required,
                                      help, metavar)
        self.values = 0

    def __call__(self, parser, args, values, option_string=None):
        """
        Procesa el valor del argumento y actualiza el contador de verbosidad.

        Args:
            parser (argparse.ArgumentParser): El objeto parser de argparse.
            args (Namespace): Espacio de nombres que contiene los argumentos.
            values (any): Valor del argumento.
            option_string (str, optional): Cadena de opción que se utilizó.
        """
        if values is None:
            self.values += 1
        else:
            try:
                self.values = int(values)
            except ValueError:
                self.values = values.count('v') + 1
        setattr(args, self.dest, self.values)

class BatchProcess:
    """
    Clase principal para manejar los argumentos y ejecutar procesos en batch.

    Esta clase se encarga de la configuración de los argumentos de entrada y
    la ejecución de la lógica principal del proceso en batch.

    Attributes:
        config (argparse.Namespace): Espacio de nombres que contiene los 
        argumentos parseados.
    """

    def __init__(self):
        """
        Inicializa la clase y parsea los argumentos de entrada.

        """
        default_from, default_until = get_default_dates()
        parser = argparse.ArgumentParser(description='Ejecución de procesos en batch.')
        parser.add_argument('-f', '--from_time', type=str, default=default_from,
                            help='Fecha de inicio de búsqueda (formato: YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('-u', '--until', type=str, default=default_until,
                            help='Fecha de finalización de búsqueda (formato: YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('-p', '--path', type=str, default=default_path,
                            help='Path to the config.yaml relevant to test the connection')
        parser.add_argument('-v', '--verbose', nargs='?', action=VAction, dest='verbose',
                            help='Nivel de verbosidad', default=0)
        self.args = parser.parse_args()

    def get_cnf(self):
        """
        Function returning the fiel where the relevant configuration is placed.

        :return: Path to the config file
        :rtype: str
        """
        return(sef.args['default_path'])

    def run(self):
        """
        Ejecuta el proceso en batch.

        Recupera los parámetros de entrada y ejecuta la lógica principal 
        del proceso.
        """
        from_time = self.config.from_time
        until_time = self.config.until
        verbose = self.config.verbose

        if verbose >= 1:
            print(f"Parámetros recibidos: --from_time {from_time}, --until {until_time}, --verbose {verbose}")

        self.main_process()

    def main_process(self):
        """
        Simula el proceso principal de ejecución.

        Este método contiene la lógica principal del proceso y puede incluir 
        operaciones adicionales según sea necesario.
        """
        verbose = self.config.verbose

        if verbose >= 1:
            print("Ejecutando el proceso principal...")

        for i in range(3):
            if verbose >= 2:
                print(f"Iteración {i} en el bucle principal.")

            if verbose >= 3:
                print(f"Objeto importante en la iteración {i}: {{'clave': 'valor'}}")

        if verbose >= 1:
            print("Proceso principal finalizado.")


    def get_default_dates(self):
        """
        Obtiene las fechas por defecto para la búsqueda.

        Calcula las fechas de ayer desde la medianoche hasta el final del día.

        :return: Una tupla que contiene dos cadenas en el formato 
                 'YYYY-MM-DD HH:MM:SS', la primera es la fecha de inicio y la segunda 
                 es la fecha de finalización.
        :rtype: tuple
        """

        today = datetime.now()
        yesterday = today - timedelta(days=1)
        self.default_from = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        self.default_until = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
        return (self.default_from.strftime('%Y-%m-%d %H:%M:%S'), \
                self.default_until.strftime('%Y-%m-%d %H:%M:%S'))

