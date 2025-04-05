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

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Random;

public class Main {
    
    public static void main(String[] args) throws Exception {
        ///////////// HASIERAKETAK /////////////
        if (args.length != 2) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output file-a (extentzio gabe)>"); // Erabilera mezua
            System.exit(0);
        }
        File dir = new File(args[0]);
        if (!dir.isDirectory() || args[1].contains(".")) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output file-a (extentzio gabe)>");
            System.exit(0);
        }
        String inputPath = args[0];
        System.out.println("Sarrerako fitxategia: " + inputPath); // Sarrerako fitxategiaren path-a
        String outputFile = args[1];
        System.out.println("Irteerako fitxategia: " + outputFile); // Irteerako fitxategiaren path-a
        Instances instances[] = null;

        // Configure netlib-java to use pure Java implementations
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        ///////////// PROZESAMENDUA /////////////
        // Datuak karpetetatik eta dokumentuetatik irakurri eta gorde
        instances = datuBilketa.getDB().bildu(inputPath, outputFile);

        // Dev-rako iragarpenak egin
        iragarri.main(instances[0], instances[1], "Dev");
 
        // Train eta Dev konbinatu Test-erako
        Instances trainDev = new Instances(instances[0]);
        trainDev.addAll(instances[2]);

        // Test-erako iragarpenak egin
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

        // 5-fold cross-validation
        Instances dataset = datuBilketa.getDB().bildu(args[0], args[1])[0]; // Example: train dataset

        // 5-fold cross-validation: SMO PolyKernel
        weka.classifiers.functions.SMO smoPolyKernel = new weka.classifiers.functions.SMO();
        smoPolyKernel.setKernel(new weka.classifiers.functions.supportVector.PolyKernel());
        performCrossValidation(smoPolyKernel, dataset, 5, "src/emaitzak/smo_poly_kernel_5cfv.txt");

        // 5-fold cross-validation: SMO RBFKernel
        weka.classifiers.functions.SMO smoRBFKernel = new weka.classifiers.functions.SMO();
        smoRBFKernel.setKernel(new weka.classifiers.
        functions.supportVector.RBFKernel());
        performCrossValidation(smoRBFKernel, dataset, 5, "src/emaitzak/smo_rbf_kernel_5cfvs.txt");

        // 5-fold cross-validation: SMO PukKernel
        weka.classifiers.functions.SMO smoPukKernel = new weka.classifiers.functions.SMO();
        smoPukKernel.setKernel(new weka.classifiers.functions.supportVector.Puk());
        performCrossValidation(smoPukKernel, dataset, 5, "src/emaitzak/smo_puk_kernel_5cfvs.txt");

        // 5-fold cross-validation: Linear Regression
        dataset = preprocessDataLR(dataset);
        dataset.deleteWithMissingClass(); // Eliminar instancias con valores faltantes en el atributo de clase
        Classifier linearRegression = new weka.classifiers.functions.LinearRegression();
        performCrossValidationWithMetrics(linearRegression, dataset, 5, "src/emaitzak/linear_regression_5cfvs.txt");
    }

    public static void accuracyKalkulatu(String[] predictionFilePaths, Instances devSet) {
        try {
            // Emaitzak gordetzeko taula sortu
            List<String[]> table = new ArrayList<>();
            table.add(new String[]{"True Value", "Linear Regression", "SMO Poly Kernel", "SMO RBF Kernel", "SMO Puk Kernel"});

            // Iragarpenak fitxategi bakoitzetik irakurri
            Map<String, List<String>> predictions = new HashMap<>();
            for (String filePath : predictionFilePaths) {
                String method = extractMethodFromFilePath(filePath);
                predictions.put(method, readPredictions(filePath));
            }

            // Accuracy kalkulatzeko aldagaiak
            int[] correctPredictions = new int[4];
            int totalInstances = devSet.numInstances();

            // Nahasmen matrizeak sortu
            int[][] linearRegressionMatrix = new int[2][2];
            int[][] smoPolyKernelMatrix = new int[2][2];
            int[][] smoRBFKernelMatrix = new int[2][2];
            int[][] smoPukKernelMatrix = new int[2][2];

            // Dev multzoko instantziak iteratu
            for (int i = 0; i < totalInstances; i++) {
                String trueValue = devSet.instance(i).stringValue(devSet.classIndex());
                String linearRegressionPrediction = predictions.getOrDefault("Linear Regression", new ArrayList<>()).get(i);
                String smoPolyKernelPrediction = predictions.getOrDefault("SMO Poly Kernel", new ArrayList<>()).get(i);
                String smoRBFKernelPrediction = predictions.getOrDefault("SMO RBF Kernel", new ArrayList<>()).get(i);
                String smoPukKernelPrediction = predictions.getOrDefault("SMO Puk Kernel", new ArrayList<>()).get(i);

                // Iragarpen zuzenak zenbatu
                if (trueValue.equalsIgnoreCase(linearRegressionPrediction)) correctPredictions[0]++;
                if (trueValue.equalsIgnoreCase(smoPolyKernelPrediction)) correctPredictions[1]++;
                if (trueValue.equalsIgnoreCase(smoRBFKernelPrediction)) correctPredictions[2]++;
                if (trueValue.equalsIgnoreCase(smoPukKernelPrediction)) correctPredictions[3]++;

                // Nahasmen matrizeen balioak jarri
                updateConfusionMatrix(linearRegressionMatrix, trueValue, linearRegressionPrediction);
                updateConfusionMatrix(smoPolyKernelMatrix, trueValue, smoPolyKernelPrediction);
                updateConfusionMatrix(smoRBFKernelMatrix, trueValue, smoRBFKernelPrediction);
                updateConfusionMatrix(smoPukKernelMatrix, trueValue, smoPukKernelPrediction);

                // Taularen errenkada gehitu
                table.add(new String[]{trueValue, linearRegressionPrediction, smoPolyKernelPrediction, smoRBFKernelPrediction, smoPukKernelPrediction});
            }

            // Taula fitxategi batean gorde
            saveTableToFile(table, "src/emaitzak/accuracy_table.txt");

            // Nahasmen matrizeak gorde
            saveConfusionMatrixToFile(linearRegressionMatrix, "Linear Regression", "src/emaitzak/linear_regression_confusion_matrix.txt");
            saveConfusionMatrixToFile(smoPolyKernelMatrix, "SMO Poly Kernel", "src/emaitzak/smo_poly_kernel_confusion_matrix.txt");
            saveConfusionMatrixToFile(smoRBFKernelMatrix, "SMO RBF Kernel", "src/emaitzak/smo_rbf_kernel_confusion_matrix.txt");
            saveConfusionMatrixToFile(smoPukKernelMatrix, "SMO Puk Kernel", "src/emaitzak/smo_puk_kernel_confusion_matrix.txt");

            // Accuracy kalkulatu eta erakutsi
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

    private static void updateConfusionMatrix(int[][] matrix, String trueValue, String predictedValue) {
        int trueIndex = trueValue.equalsIgnoreCase("pos") ? 0 : 1;
        int predictedIndex = predictedValue.equalsIgnoreCase("pos") ? 0 : 1;
        matrix[trueIndex][predictedIndex]++;
    }

    private static void saveConfusionMatrixToFile(int[][] matrix, String modelName, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("Confusion Matrix for " + modelName + ":\n");
            writer.write("\t\tPredicted: pos\tPredicted: neg\n");
            writer.write("True: pos\t" + matrix[0][0] + "\t\t" + matrix[0][1] + "\n");
            writer.write("True: neg\t" + matrix[1][0] + "\t\t" + matrix[1][1] + "\n");
            writer.write("-----------------------------------\n");
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
                    // Iragarpen klasea lerrotik atera
                    String[] parts = line.split("k:"); // Lerroa "k:"-ren arabera zatitu
                    if (parts.length > 1) {
                        String predictedClass = parts[1].trim(); // Iragarpen klasea lortu
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

    public static void performCrossValidation(Classifier model, Instances dataset, int folds, String outputFilePath) throws Exception {
        dataset.setClassIndex(0);
        dataset.deleteWithMissingClass();

        // Ebaluzio objektua sortu
        Evaluation evaluation = new Evaluation(dataset);

        // 5-fold cross-validation exekutatu
        evaluation.crossValidateModel(model, dataset, folds, new Random(1));

        // Emaitzak gorde
        StringBuilder results = new StringBuilder();
        results.append("=== Cross-validation Results ===\n");
        results.append("Model: ").append(model.getClass().getSimpleName()).append("\n");
        results.append("Accuracy: ").append(String.format("%.2f", evaluation.pctCorrect())).append("%\n");
        results.append("Precision: ").append(String.format("%.2f", evaluation.weightedPrecision())).append("\n");
        results.append("Recall: ").append(String.format("%.2f", evaluation.weightedRecall())).append("\n");
        results.append("F1 Score: ").append(String.format("%.2f", evaluation.weightedFMeasure())).append("\n");
        results.append(evaluation.toMatrixString("Confusion Matrix")).append("\n");

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            writer.write(results.toString());
        }
    }

    public static void performCrossValidationWithMetrics(Classifier model, Instances dataset, int folds, String outputFilePath) throws Exception {
        // Linear Regression-erako berezia
        dataset.setClassIndex(0);

        // Ebaluzio objektua sortu
        Evaluation evaluation = new Evaluation(dataset);

        // Ejecutar 5-fold cross-validation
        evaluation.crossValidateModel(model, dataset, folds, new Random(1));

        // Kalkuluetarako aldagaiak
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        int trueNegatives = 0;
        for (int i = 0; i < dataset.numInstances(); i++) {
            double trueValue = dataset.instance(i).classValue();
            double predictedValue = evaluation.predictions().get(i).predicted();

            String predictedClass = predictedValue >= 0.5 ? "pos" : "neg";
            String trueClass = trueValue == 1.0 ? "pos" : "neg";

            if (trueClass.equals("pos")) {
                if (predictedClass.equals("pos")) {
                    truePositives++;
                } else {
                    falseNegatives++;
                }
            } else if (trueClass.equals("neg")) {
                if (predictedClass.equals("neg")) {
                    trueNegatives++;
                } else {
                    falsePositives++;
                }
            }
        }

        // Balioak kalkulatu
        double precision = (double) truePositives / (truePositives + falsePositives);
        double recall = (double) truePositives / (truePositives + falseNegatives);
        double f1Score = 2 * ((precision * recall) / (precision + recall));
        double accuracy = (double) (truePositives + trueNegatives) / dataset.numInstances();

        // Emaitzak gorde
        StringBuilder results = new StringBuilder();
        results.append("=== Cross-validation Results ===\n");
        results.append("Model: ").append(model.getClass().getSimpleName()).append("\n");
        results.append("Accuracy: ").append(String.format("%.2f", accuracy * 100)).append("%\n");
        results.append("Precision: ").append(String.format("%.4f", precision)).append("\n");
        results.append("Recall: ").append(String.format("%.4f", recall)).append("\n");
        results.append("F1 Score: ").append(String.format("%.4f", f1Score)).append("\n");
        results.append("Confusion Matrix\n");
        results.append("   a   b   <-- classified as\n");
        results.append(String.format("  %3d %3d |   a = pos\n", truePositives, falseNegatives));
        results.append(String.format("  %3d %3d |   b = neg\n", falsePositives, trueNegatives));

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            writer.write(results.toString());
        }
    }

    private static Instances preprocessDataLR(Instances dataset) {
        try {
            dataset.setClassIndex(0);
            int classIndex = dataset.classIndex();

            if (dataset.attribute(classIndex).isNominal() && dataset.attribute(classIndex).numValues() == 2) {
                ArrayList<Attribute> attributes = new ArrayList<>();
                for (int i = 0; i < dataset.numAttributes(); i++) {
                    if (i == classIndex) {
                        attributes.add(new Attribute("class"));
                    } else {
                        attributes.add(dataset.attribute(i));
                    }
                }

                Instances newDataset = new Instances(dataset.relationName(), attributes, dataset.numInstances());
                newDataset.setClassIndex(classIndex);

                // Convertir las instancias
                for (int i = 0; i < dataset.numInstances(); i++) {
                    DenseInstance newInstance = new DenseInstance(newDataset.numAttributes());
                    newInstance.setDataset(newDataset);

                    for (int j = 0; j < dataset.numAttributes(); j++) {
                        if (j == classIndex) {
                            String classValue = dataset.instance(i).stringValue(classIndex);
                            if (classValue.equalsIgnoreCase("pos")) {
                                newInstance.setValue(classIndex, 1.0);
                            } else if (classValue.equalsIgnoreCase("neg")) {
                                newInstance.setValue(classIndex, 0.0);
                            } else {
                                newInstance.setMissing(classIndex);
                            }
                        } else {
                            newInstance.setValue(j, dataset.instance(i).value(j));
                        }
                    }

                    newDataset.add(newInstance);
                }

                return newDataset;
            }

            return dataset;
        } catch (Exception e) {
            System.out.println("Error al procesar los datos para Linear Regression.");
            e.printStackTrace();
            return null;
        }
    }
}
