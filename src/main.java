package src;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

@SuppressWarnings("all")
public class main {
    public static void main(String[] args) throws Exception {
        ///////////// HASIERAKETAK /////////////
        if (args.length != 2) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output file-a (extentzio gabe)>");
            System.exit(0);
        }
        File dir = new File(args[0]);
        if (!dir.isDirectory() || args[1].contains(".")) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output file-a (extentzio gabe)>");
            System.exit(0);
        }
        String inputPath = args[0];
        System.out.println("Sarrerako fitxategia: " + inputPath);
        String outputFile = args[1];
        System.out.println("Irteerako fitxategia: " + outputFile);
        Instances instances[] = null;

        // Configurar netlib-java para usar implementaciones en Java puro
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        ///////////// PROCESAMENDUA /////////////
        instances = datuBilketa.getDB().bildu(inputPath, outputFile);

        // Realizar predicciones para Dev
        iragarri.main(instances[0], instances[1], "Dev");

        // Combinar Train y Dev para Test
        Instances trainDev = new Instances(instances[0]);
        trainDev.addAll(instances[2]);

        // Realizar predicciones para Test
        iragarri.main(trainDev, instances[3], "Test");

        ///////////// ACCURACY KALKULATU /////////////
        File emaitzakDir = new File("src/emaitzak");
        if (emaitzakDir.isDirectory()) {
            File[] predictionFiles = emaitzakDir.listFiles((dir1, name) -> name.startsWith("iragarpena_Dev") && name.endsWith(".txt"));
            if (predictionFiles != null) {
                String[] predictionFilePaths = new String[predictionFiles.length];
                for (int i = 0; i < predictionFiles.length; i++) {
                    predictionFilePaths[i] = predictionFiles[i].getAbsolutePath();
                }
                accuracyKalkulatu(predictionFilePaths, instances[2]);
            }
        }

        System.exit(0);

        ///////////// IDAZKETA /////////////
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances[0]);
        try {
            saver.setFile(new File(outputFile));
            saver.writeBatch();
            System.out.println("Datuak gorde dira: " + outputFile);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da " + outputFile + " fitxategia gorde.");
            e.printStackTrace();
        }
    }

    public static void accuracyKalkulatu(String[] predictionFilePaths, Instances devSet) {
        try {
            // Crear una tabla para almacenar los resultados
            List<String[]> table = new ArrayList<>();
            table.add(new String[]{"True Value", "Linear Regression", "SMO Poly Kernel", "SMO RBF Kernel", "SMO Puk Kernel"});

            // Leer las predicciones de cada archivo
            Map<String, List<String>> predictions = new HashMap<>();
            for (String filePath : predictionFilePaths) {
                String method = extractMethodFromFilePath(filePath);
                predictions.put(method, readPredictions(filePath));
            }

            // Variables para calcular accuracy
            int[] correctPredictions = new int[4];
            int totalInstances = devSet.numInstances();

            // Iterar sobre las instancias del conjunto Dev
            for (int i = 0; i < totalInstances; i++) {
                String trueValue = devSet.instance(i).stringValue(devSet.classIndex());
                String linearRegressionPrediction = predictions.getOrDefault("Linear Regression", new ArrayList<>()).get(i);
                String smoPolyKernelPrediction = predictions.getOrDefault("SMO Poly Kernel", new ArrayList<>()).get(i);
                String smoRBFKernelPrediction = predictions.getOrDefault("SMO RBF Kernel", new ArrayList<>()).get(i);
                String smoPukKernelPrediction = predictions.getOrDefault("SMO Puk Kernel", new ArrayList<>()).get(i);

                // Contar predicciones correctas
                if (trueValue.equalsIgnoreCase(linearRegressionPrediction)) correctPredictions[0]++;
                if (trueValue.equalsIgnoreCase(smoPolyKernelPrediction)) correctPredictions[1]++;
                if (trueValue.equalsIgnoreCase(smoRBFKernelPrediction)) correctPredictions[2]++;
                if (trueValue.equalsIgnoreCase(smoPukKernelPrediction)) correctPredictions[3]++;

                // Añadir la fila a la tabla
                table.add(new String[]{trueValue, linearRegressionPrediction, smoPolyKernelPrediction, smoRBFKernelPrediction, smoPukKernelPrediction});
            }

            // Guardar la tabla en un archivo
            saveTableToFile(table, "src/emaitzak/accuracy_table.txt");

            // Calcular y mostrar accuracies
            System.out.println("Accuracy Results:");
            System.out.println("Linear Regression: " + String.format("%.2f", (double) correctPredictions[0] / totalInstances * 100) + "%");
            System.out.println("SMO Poly Kernel: " + String.format("%.2f", (double) correctPredictions[1] / totalInstances * 100) + "%");
            System.out.println("SMO RBF Kernel: " + String.format("%.2f", (double) correctPredictions[2] / totalInstances * 100) + "%");
            System.out.println("SMO Puk Kernel: " + String.format("%.2f", (double) correctPredictions[3] / totalInstances * 100) + "%");

            System.out.println("Accuracy table saved to: src/emaitzak/accuracy_table.txt");
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da accuracy taula sortu.");
            e.printStackTrace();
        }
    }

    private static String extractMethodFromFilePath(String filePath) {
        if (filePath.contains("LinearRegression")) return "Linear Regression";
        if (filePath.contains("SMO_PolyKernel")) return "SMO Poly Kernel";
        if (filePath.contains("SMO_RBFKernel")) return "SMO RBF Kernel";
        if (filePath.contains("SMO_PukKernel")) return "SMO Puk Kernel";
        return "Unknown";
    }

    private static List<String> readPredictions(String filePath) throws IOException {
        List<String> predictions = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.contains("instantzia:")) {
                    // Extraer la clase predicha desde la línea
                    String[] parts = line.split("k:"); // Dividir la línea por "k:"
                    if (parts.length > 1) {
                        String predictedClass = parts[1].trim(); // Obtener la clase predicha
                        predictions.add(predictedClass);
                    }
                }
            }
        }
        return predictions;
    }

    private static void saveTableToFile(List<String[]> table, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String[] row : table) {
                writer.write(String.join("\t", row));
                writer.newLine();
            }
        }
    }
}
