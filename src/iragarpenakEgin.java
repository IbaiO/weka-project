package src;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class iragarpenakEgin {
    private static void main(Instances dataset) throws Exception {
        LinearRegression modelLR = linearRegression.linearRegressionSortu(null);
        SMO modelSMO = FSSetaSMO.main(null);
        if (modelLR != null || modelSMO != null) {
            iragarri(modelLR, dataset);
        }
        konprobatuDev();
    }
    
    private static void iragarri(LinearRegression model, Instances dataset) {
        String outputFilePath = "/home/ibai/GitHub/weka-project/probaData/iragarpena_LinearRegression.txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            // Write predictions
            for (int i = 0; i < dataset.numInstances(); i++) {
                Instance instance = dataset.instance(i);
                double prediction = model.classifyInstance(instance);
                String predictedClass = prediction > 0.5 ? "Pos" : "Neg";

                // Format attributes
                StringBuilder attributes = new StringBuilder();
                for (int j = 0; j < instance.numAttributes(); j++) {
                    if (j != instance.classIndex()) { // Skip the class attribute
                        attributes.append(instance.value(j));
                        if (j < instance.numAttributes() - 1) {
                            attributes.append(", ");
                        }
                    }
                }

                // Write formatted line
                writer.write((i + 1) + ". instantzia: " + predictedClass + ", (" + attributes + ")\n");
            }
            System.out.println("Predictions saved to: " + outputFilePath);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
            e.printStackTrace();
        }
    }

    private static void konprobatuDev() {
        System.out.println("Konprobatu dev");
    }
}