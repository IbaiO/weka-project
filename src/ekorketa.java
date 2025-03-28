package src;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;

import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("all")
public class ekorketa {

    private static ekorketa nireDG = null;

    public static ekorketa getEkorketa() {
        if (nireDG == null) {
            nireDG = new ekorketa();
        }
        return nireDG;
    }

    public Instances ekorketa(String inPath) {
        String outPath = null; //TODO: Zein izenarekin aterako da .arff?
        Instances data = null;
        try {
            // Leer las carpetas dentro del directorio
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("El path proporcionado no es un directorio.");
            }

            // Obtener las carpetas (clases)
            File[] classFolders = dir.listFiles(File::isDirectory);
            if (classFolders == null || classFolders.length == 0) {
                throw new Exception("No se encontraron carpetas dentro del directorio.");
            }

            // Crear atributos para el archivo .arff
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_name", (List<String>) null)); // Nombre del archivo
            List<String> classValues = new ArrayList<>();
            for (File folder : classFolders) {
                classValues.add(folder.getName()); // Nombres de las carpetas como clases
            }
            attributes.add(new Attribute("class", classValues)); // Atributo de clase

            // Crear el conjunto de datos
            data = new Instances("Dataset", attributes, 0);
            data.setClassIndex(1); // El atributo "class" es el atributo de clase

            // Recorrer las carpetas y agregar instancias
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        DenseInstance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), file.getName()); // Nombre del archivo
                        instance.setValue(attributes.get(1), folder.getName()); // Clase (nombre de la carpeta)
                        data.add(instance);
                    }
                }
            }

            // Guardar el archivo .arff
            FileWriter writer = new FileWriter(outPath);
            writer.write(data.toString());
            writer.close();

            System.out.println("Archivo .arff creado exitosamente en: " + outPath);

        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
        }
        return data;
    }
}
