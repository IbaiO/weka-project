package src;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

import java.util.ArrayList;

public class linearRegression {
    
    public static LinearRegression main(Instances dataset) throws Exception {
        // Klasearen indizea ezarri.
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1); // Set the last attribute as the class
        }

        // Datuak prozesatu.
        dataset = preprocessData(dataset);
        if (dataset == null) {
            System.out.println("ERROREA: Ezin izan dira datuak prozesatu."); // Error: Could not process data
            return null;
        }

        // Modeloa eraiki.
        LinearRegression model = buildModel(dataset);
        if (model == null) {
            System.out.println("ERROREA: Ezin izan da modeloa eraiki."); // Error: Could not build the model
            return null;
        }
        return model;
    }

    private static Instances preprocessData(Instances dataset) {
        try {
            dataset.setClassIndex(0);
            // Atributu bitar nominalak zenbakizkotan bihurtu
            if (dataset.classIndex() != -1) {
                int classIndex = dataset.classIndex();
                if (dataset.attribute(classIndex).isNominal() && dataset.attribute(classIndex).numValues() == 2) {
                    // Atributu zerrenda bat sortu, DataSet berriarentzat
                    ArrayList<Attribute> attributes = new ArrayList<>();
                    for (int i = 0; i < dataset.numAttributes(); i++) {
                        if (i == classIndex) {
                            // Klase atributu nominala atributu zenbakizko batekin ordezkatu
                            attributes.add(new Attribute("class"));
                        } else {
                            attributes.add(dataset.attribute(i));
                        }
                    }

                    // DataSet berria sortu Pos/Neg klasea zenbaki gisa jarrita
                    // Pos: 1, Neg: 0
                    Instances newDataset = new Instances(dataset.relationName(), attributes, dataset.numInstances());
                    newDataset.setClassIndex(classIndex);
                    // Instantziak DataSet berrira gehitu
                    for (int i = 0; i < dataset.numInstances(); i++) {
                        DenseInstance newInstance = new DenseInstance(newDataset.numAttributes());
                        newInstance.setDataset(newDataset);

                        for (int j = 0; j < dataset.numAttributes(); j++) {
                            if (j == classIndex) {
                                String classValue = dataset.instance(i).stringValue(classIndex);
                                if (classValue.equalsIgnoreCase("pos")) {
                                    newInstance.setValue(classIndex, 1); // Pos klasea 1 bezala ezarri
                                } else if (classValue.equalsIgnoreCase("neg")) {
                                    newInstance.setValue(classIndex, 0); // Neg klasea 0 bezala ezarri
                                } else if (classValue.equals("?")) {
                                    newInstance.setMissing(classIndex); // Balio ezezaguna
                                } else {
                                    newInstance.setValue(classIndex, -1); // Balio ezezaguna (-1)
                                }
                            } else {
                                newInstance.setValue(j, dataset.instance(i).value(j));
                            }
                        }

                        newDataset.add(newInstance);
                    }
                    dataset = newDataset;
                }
            }

            return dataset;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan dira datuak prozesatu."); // Error: Could not process data
            e.printStackTrace();
            return null;
        }
    }

    private static LinearRegression buildModel(Instances dataset) {
        try {
            LinearRegression model = new LinearRegression();
            model.buildClassifier(dataset); // Build the Linear Regression model
            return model;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da modeloa eraiki."); // Error: Could not build the model
            e.printStackTrace();
            return null;
        }
    }
}