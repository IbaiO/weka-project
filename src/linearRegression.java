package src;

import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class linearRegression {
    public static LinearRegression main(Instances dataset) throws Exception {
        /*
        String dataSource = args[0];

        Instances dataset = loadData(dataSource);
        if (dataset == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return null;
        }
        */
        dataset.setClassIndex(dataset.numAttributes() - 1);
        dataset = preprocessData(dataset);

        LinearRegression model = buildModel(dataset);
        if (model != null)
            return model;
        return null;
    }

    private static Instances preprocessData(Instances dataset) {
        try {
            // Check if there are string attributes
            boolean hasStringAttributes = false;
            for (int i = 0; i < dataset.numAttributes(); i++) {
                if (dataset.attribute(i).isString()) {
                    hasStringAttributes = true;
                    break;
                }
            }

            if (hasStringAttributes) {
                // Apply StringToWordVector filter
                StringToWordVector stringToWordVector = new StringToWordVector();
                stringToWordVector.setInputFormat(dataset);
                dataset = Filter.useFilter(dataset, stringToWordVector);
            }

            // Check if the class attribute is binary and nominal
            if (dataset.classIndex() != -1) {
                int classIndex = dataset.classIndex();
                if (dataset.attribute(classIndex).isNominal() && dataset.attribute(classIndex).numValues() == 2) {
                    System.out.println("Binary class attribute detected. Converting it to numeric.");
                    for (int i = 0; i < dataset.numInstances(); i++) {
                        String classValue = dataset.instance(i).stringValue(classIndex);
                        // Map binary class values to numeric (e.g., "yes" -> 1, "no" -> 0)
                        if (classValue.equalsIgnoreCase("pos")) {
                            dataset.instance(i).setValue(classIndex, 1);
                        } else if (classValue.equalsIgnoreCase("neg")) {
                            dataset.instance(i).setValue(classIndex, 0);
                        } else {
                            System.out.println("ERROREA: Klase balio ezezaguna aurkitu da: " + classValue);
                            return null; // Return null if an unknown class value is found
                        }
                    }
                }
            }

            // Binary attributes are already numeric (0 or 1), so no additional processing is needed
            return dataset;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datuak aurreprozesatu.");
            e.printStackTrace();
            return null;
        }
    }

    private static LinearRegression buildModel(Instances dataset) {
        try {
            LinearRegression model = new LinearRegression();
            dataset.setClassIndex(dataset.numAttributes() - 1); // Set the class attribute
            model.buildClassifier(dataset);
            return model;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da modeloa eraiki.");
            e.printStackTrace();
            return null;
        }
    }
}