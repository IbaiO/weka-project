package src;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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
        String inputPath = args[0]; System.out.println("Sarrerako fitxategia: " + inputPath);
        String outputFile = args[1]; System.out.println("Irteerako fitxategia: " + outputFile);
        Instances instances[] = null;
        
        // Configurar netlib-java para usar implementaciones en Java puro
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        ///////////// PROCESAMENDUA /////////////
        instances = datuBilketa.getDB().bildu(inputPath, outputFile);
        iragarri.main(instances[0], instances[2], "Dev");
        Instances trainDev = new Instances(instances[0]);
        trainDev.addAll(instances[1]);
        iragarri.main(trainDev, instances[3], "Test");
        
        ///////////// ACCURACY KALKULATU /////////////
        File emaitzakDir = new File("src/emaitzak");
        if (emaitzakDir.isDirectory()) {
            File[] predictionFiles = emaitzakDir.listFiles((dir1, name) -> name.startsWith("iragarpena_Dev") && name.endsWith(".txt"));
            if (predictionFiles != null) {
                for (File predictionFile : predictionFiles) {
                    System.out.println("Calculating accuracy for: " + predictionFile.getName());
                    accuracyKalkulatu(predictionFile.getAbsolutePath(), instances[1]);
                }
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

    public static double accuracyKalkulatu(String predictionsFilePath, Instances devSet) {
        try (BufferedReader reader = new BufferedReader(new FileReader(predictionsFilePath))) {
            int correctPredictions = 0;
            int totalInstances = devSet.numInstances();
            int instanceIndex = 0;

            String line;
            while ((line = reader.readLine()) != null) {
                // Skip lines that do not contain predictions
                if (!line.contains("instantzia:")) {
                    continue;
                }

                // Extract the predicted class from the line
                String predictedClass = line.trim().endsWith("Pos") ? "pos" : "neg";

                // Get the actual class from the Dev dataset
                String actualClass = devSet.instance(instanceIndex).stringValue(devSet.classIndex());

                // Compare the predicted class with the actual class
                if (predictedClass.equalsIgnoreCase(actualClass)) {
                    correctPredictions++;
                }

                instanceIndex++;
            }

            // Calculate accuracy as a percentage
            double accuracy = (double) correctPredictions / totalInstances * 100;
            System.out.println("Accuracy: " + String.format("%.2f", accuracy) + "%");
            return accuracy;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da iragarpen fitxategia irakurri edo prozesatu.");
            e.printStackTrace();
            return -1;
        }
    }
}
