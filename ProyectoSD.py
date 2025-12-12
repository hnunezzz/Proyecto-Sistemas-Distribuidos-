import os
import sys
import numpy as np
import warnings

# CONFIGURACIÓN E IMPORTACIONES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_DIR = os.path.join(BASE_DIR, "books")
VECTORES_DIR = os.path.join(BASE_DIR, "vectores_tfidf")
MATRIX_DIR = os.path.join(BASE_DIR, "matriz_similitud")
MATRIX_FILE = os.path.join(MATRIX_DIR, "matriz.npy")

warnings.filterwarnings("ignore")


# BACKEND (Procesamiento de Texto y Generación de Matriz)
# Esto es lo que se ejecuta solo con la bandera --setup

def ejecutar_setup():
    print(" INICIANDO MODO SETUP (Procesamiento Masivo) \n")

    # Importaciones pesadas solo si son necesarias
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml import Pipeline
    from pyspark.sql.functions import input_file_name, collect_list, concat_ws, col, udf
    from pyspark.sql.types import ArrayType, FloatType
    from pyspark.ml.linalg import SparseVector, DenseVector
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd

    # 1. VALIDACIÓN
    ruta_libros = os.path.join(BOOKS_DIR, "*.txt")
    if not os.path.exists(BOOKS_DIR):
        print(f" ERROR CRÍTICO: No existe la carpeta {BOOKS_DIR}")
        return

    # 2. INICIAR SPARK
    spark = SparkSession.builder \
        .appName("SistemaRecomendacion_Setup") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print(" [1/5] Leyendo libros y limpiando texto...")
    try:
        df = spark.read.text(ruta_libros).withColumn("filename", input_file_name())
    except Exception as e:
        print(f" Error leyendo archivos: {e}")
        spark.stop(); return

    # Agrupar texto por archivo
    df = df.groupBy("filename").agg(collect_list("value").alias("texto"))
    df = df.withColumn("texto", concat_ws(" ", "texto"))

    # 3. PIPELINE NLP
    print(" [2/5] Ejecutando Pipeline NLP (Tokenización -> StopWords -> TF-IDF)...")
    tokenizer = Tokenizer(inputCol="texto", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_limpios")
    hashingTF = HashingTF(inputCol="tokens_limpios", outputCol="tf", numFeatures=20000)
    idf = IDF(inputCol="tf", outputCol="tfidf")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    model = pipeline.fit(df)
    df_transformed = model.transform(df)

    # 4. CONVERSIÓN A MATRIZ
    print(" [3/5] Preparando vectores para cálculo matemático...")
    
    # UDF para convertir vectores de Spark a listas de Python
    def sparse_to_array(v):
        if isinstance(v, SparseVector): return DenseVector(v).tolist()
        elif isinstance(v, DenseVector): return v.tolist()
        return []

    to_array_udf = udf(sparse_to_array, ArrayType(FloatType()))
    df_vectores = df_transformed.select("filename", "tfidf") \
        .withColumn("vector", to_array_udf(col("tfidf")))

    # Convertir a Pandas para usar Scikit-Learn
    pdf = df_vectores.select("filename", "vector").toPandas()
    
    # Ordenar por nombre de archivo para que los índices coincidan siempre
    # Extraemos solo el nombre del archivo de la ruta completa por ejemplo: "file:///.../11.txt" -> "11.txt"
    pdf['clean_filename'] = pdf['filename'].apply(lambda x: os.path.basename(x))
    pdf = pdf.sort_values('clean_filename').reset_index(drop=True)

    print(f" [4/5] Calculando similitud del coseno entre {len(pdf)} libros...")
    matriz = cosine_similarity(pdf["vector"].tolist())

    # 5. GUARDADO
    print(f" [5/5] Guardando matriz en {MATRIX_FILE}...")
    os.makedirs(MATRIX_DIR, exist_ok=True)
    np.save(MATRIX_FILE, matriz)

    spark.stop()
    print("\n¡SETUP COMPLETADO EXITOSAMENTE!")
    print(" Ahora puedes pedir recomendaciones ejecutando: python sistema_completo.py <nombre_libro>")






# FRONTEND (Consultas y Resúmenes)
# Esto es lo que se ejecuta solo con el argumento que le pasaremos al codigo usando el numero del libro.txt
def obtener_titulo(nombre_archivo):
    """Extrae título heurísticamente."""
    ruta = os.path.join(BOOKS_DIR, nombre_archivo)
    try:
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            lineas = f.readlines()
    except: return nombre_archivo

    for linea in lineas[:200]: # Buscar Title:
        if linea.strip().lower().startswith("title:"):
            return linea.split(":", 1)[1].strip()
    
    for linea in lineas[:150]: # Buscar primera línea válida
        l = linea.strip()
        if len(l) > 3 and not l.startswith("***") and not l.isupper():
            return l
    return nombre_archivo




def ejecutar_consulta(libro_input):
    from transformers import pipeline

    # 1. VERIFICACIONES DE EXISTENCIA DE LOS LIBROS EN LA CARPETA BOOKS
    if not os.path.exists(MATRIX_FILE):
        print("\n [!] ERROR: No se encontró la matriz de similitud.")
        print("     Debes ejecutar primero el setup: python3 sistema_completo.py --setup")
        return

    libros_disponibles = sorted([f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt")])
    
    if libro_input not in libros_disponibles:
        print(f"\n El libro '{libro_input}' no existe en la carpeta books/.")
        return




    # 2. RECOMENDACIÓN
    print(f"\n\nRECOMENDACIONES PARA: {obtener_titulo(libro_input)}")
    
    matriz = np.load(MATRIX_FILE)
    idx = libros_disponibles.index(libro_input)
    similitudes = matriz[idx]

    orden = np.argsort(similitudes)[::-1]
    top_10 = orden[1:11] 

    for i, idx_sim in enumerate(top_10, start=1):
        archivo = libros_disponibles[idx_sim]
        print(f"{i}. {archivo} — {obtener_titulo(archivo)} (Similitud: {similitudes[idx_sim]:.3f})")




    # 3. RESUMEN DE 20 PALABRAS
    print(f" \n\nGenerando resumen de 20 palabras para: {libro_input}...")
    
    try:
        ruta_libro = os.path.join(BOOKS_DIR, libro_input)
        with open(ruta_libro, "r", encoding="utf-8", errors="ignore") as f:
            texto_completo = f.read()


        # Buscamos la marca donde empieza el libro real para saltarnos la licencia y titulos de los libros
        marcador = "*** START OF"
        if marcador in texto_completo:
            # Nos quedamos solo con lo que está DESPUÉS del marcador
            texto_real = texto_completo.split(marcador)[-1]
        else:
            texto_real = texto_completo
        
        # Saltamos un poco más (500 caracteres) para evitar títulos repetidos y tomamos un bloque grande
        texto_para_ia = texto_real[500:3500]
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Generamos el resumen
        resumen_raw = summarizer(texto_para_ia, max_length=60, min_length=30, do_sample=False)[0]["summary_text"]
        
        resumen_final = " ".join(resumen_raw.split()[:20]) # Cortar a 20 palabras
        print(f"\nRESUMEN:\n\"{resumen_final}...\"\n")
        
    except Exception as e:
        print(f" Error generando resumen: {e}")



# MAIN
if __name__ == "__main__":

    argumento = sys.argv[1]

    if argumento == "--setup":
        ejecutar_setup()
    else:
        ejecutar_consulta(argumento)