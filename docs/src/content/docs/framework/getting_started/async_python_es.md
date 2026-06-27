---
Titulo: Programacion Asincronica en python
---

Si tu eres nuevo en la Programacion Asincronica en Python, esta Pagina es para ti

En la LlamaIndex, en concreto, muchas operaciones y funciones admiten la ejecucion asincrona. Esto permite ejecutar varias operaciones simultaneamente sin bloquear el hilo principal, lo que contribuye a mejorar el rendimiento general en muchos casos.

Estos son algunos de los conceptos clave que debes comprender:

## 1. Conceptos Basicos de `asyncio`

- **Bucle de Eventos**:
  El bucle de eventos gestiona la planificacion y ejecucion de operaciones asincronas. Comprueba y ejecuta continuamente tareas(corutinas). Todas las operaciones asincronas se ejecutan mediante este bucle y solo puede haber un bucle de eventos por hilo.

- **`asyncio.run()`**:
  Esta Funcion es el punto de entrada para ejecutar un programa asincronico. Crea y administra el bucle de eventos y lo limpia una vez que se completa. Recuerde que esta diseñado para ser llamado una vez por hilo. Algunos marcos como FastAPI ejecutaran el bucle de eventos por usted, otros requeriran que lo ejecute usted mismo

## 2. Funciones asíncronas y `await`

* **Definir funciones asíncronas**:
  Usa la sintaxis `async def` para definir una función asíncrona, también llamada corrutina. En lugar de ejecutarse inmediatamente, llamar a una función asíncrona devuelve un objeto corrutina que debe ser programado y ejecutado.

* **Usar `await`**:
  Dentro de una función asíncrona, `await` se usa para pausar la ejecución de esa función hasta que la tarea esperada se complete. Cuando escribes `await some_fn()`, la función cede el control de nuevo al event loop para que otras tareas puedan ser programadas y ejecutadas. Solo una función asíncrona se ejecuta a la vez, y cooperan entre sí cediendo el control con `await`.

## 3. Explicación de la concurrencia

* **Concurrencia cooperativa**:
  Aunque puedes programar múltiples tareas asíncronas, solo una tarea se ejecuta a la vez. Esto es diferente del paralelismo real, donde varias tareas se ejecutan al mismo tiempo. Cuando una tarea llega a un `await`, suspende su ejecución para que otra tarea pueda ejecutarse. Esto hace que los programas asíncronos sean excelentes para tareas limitadas por I/O, donde es común esperar, como llamadas a APIs de LLMs y otros servicios.

* **No es paralelismo real**:
  `asyncio` permite concurrencia, pero no ejecuta tareas en paralelo. Para trabajo limitado por CPU que requiere ejecución en paralelo, considera usar threading o multiprocessing. LlamaIndex normalmente evita multiprocessing en la mayoría de los casos y deja que el usuario lo implemente, ya que hacerlo de una manera segura y eficiente puede ser complejo.

## 4. Manejo de código bloqueante (síncrono)

* **`asyncio.to_thread()`**:
  A veces necesitas ejecutar código síncrono, es decir, bloqueante, sin congelar tu programa asíncrono. `asyncio.to_thread()` envía ese código bloqueante a un hilo separado, permitiendo que el event loop continúe procesando otras tareas. Úsalo con cuidado, ya que añade cierta sobrecarga y puede hacer que la depuración sea más difícil.

* **Alternativa: Executors**:
  También puedes encontrar el uso de `loop.run_in_executor()` para manejar funciones bloqueantes.

## 5. Un ejemplo práctico

A continuación se muestra un ejemplo que demuestra cómo escribir y ejecutar funciones asíncronas con `asyncio`:

```python
import asyncio


async def fetch_data(delay):
    print(f"Started fetching data with {delay}s delay")

    # Simula trabajo limitado por I/O, como una operación de red
    await asyncio.sleep(delay)

    print("Finished fetching data")
    return f"Data after {delay}s"


async def main():
    print("Starting main")

    # Programa dos tareas de forma concurrente
    task1 = asyncio.create_task(fetch_data(2))
    task2 = asyncio.create_task(fetch_data(3))

    # Espera hasta que ambas tareas se completen
    result1, result2 = await asyncio.gather(task1, task2)

    print(result1)
    print(result2)
    print("Main complete")


if __name__ == "__main__":
    asyncio.run(main())
```