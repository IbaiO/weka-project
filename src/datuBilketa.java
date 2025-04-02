package src;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;

import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("all")
public class datuBilketa {

    private static datuBilketa nireDB = null;

    public static datuBilketa getDB() {
        if (nireDB == null) {
            nireDB = new datuBilketa();
        }
        return nireDB;
    }

    public Instances[] bildu(String inPath, String outFile) {
        // Sarrerako datuak irakurri
        Instances train = bilduTrain (inPath + "/train", outFile);
        save(train, outFile + "Train.arff"); // Gorde datuak
        // Aldatu BoW formatura
        Instances trainBoW = NonSparseBoW.getNonSparseBoW().transformTrain(train);
        save(trainBoW, outFile + "TrainBoW.arff"); // Gorde datuak

        // Atera atributuak
        String[] attributes = new String[trainBoW.numAttributes()];
        for (int i = 0; i < trainBoW.numAttributes(); i++) {
            attributes[i] = trainBoW.attribute(i).name();
        }
        // Aukeratutako atributuak gorde
        save(attributes, outFile + "attributes.txt"); // Gorde datuak


        // Ateratako atributuak Dev-ri pasatu
        Instances dev = bilduDevTest (inPath + "/dev", attributes);
        save(dev, outFile + "Dev.arff"); // Gorde datuak
        Instances devBoW = NonSparseBoW.getNonSparseBoW().transformDevTest(dev);
        save(dev, outFile + "DevBoW.arff"); // Gorde datuak


        // Ateratako atributuak Test-ri pasatu
        Instances test = bilduDevTest (inPath + "/test", attributes);
        save(test, outFile + "Test.arff"); // Gorde datuak
        Instances testBoW = NonSparseBoW.getNonSparseBoW().transformDevTest(test);
        save(testBoW, outFile + "TestBoW.arff"); //Gorde datuak
        
        return new Instances[] {train, dev, test};        
    }

    private Instances bilduTrain(String inPath, String outFile) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // Karpetak (klasea) lortu
            File[] classFolders = dir.listFiles(File::isDirectory);

            // .arff fitxategirako atributuak sortu
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_content", (List<String>) null)); // Fitxategiaren edukia
            List<String> classValues = new ArrayList<>();
            for (File folder : classFolders) {
                classValues.add(folder.getName()); // Karpeten izenak klase bezala
            }
            attributes.add(new Attribute("class", classValues)); // Klase atributua

            // Datu multzoa sortu
            Instances dataTrain = new Instances("Dataset", attributes, 0);
            dataTrain.setClassIndex(1); // "class" atributua klase atributua da

            // Karpetak zeharkatu eta instantziak gehitu
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        // Fitxategiaren edukia irakurri
                        StringBuilder contentBuilder = new StringBuilder();
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                contentBuilder.append(line).append("\n");
                            }
                        } catch (Exception e) {
                            System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                            e.printStackTrace();
                        }

                        // Instantzia sortu eta datuak gehitu
                        DenseInstance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), contentBuilder.toString()); // Fitxategiaren edukia
                        instance.setValue(attributes.get(1), folder.getName()); // Klasea (karpetaren izena)
                        dataTrain.add(instance);
                    }
                }
            }

            return dataTrain;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private Instances bilduDevTest(String inPath, String[] attributes) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // Karpetak (klasea) lortu
            File[] classFolders = dir.listFiles(File::isDirectory);

            // Atributuak sortu
            ArrayList<Attribute> attributeList = new ArrayList<>();
            for (String attribute : attributes) {
                attributeList.add(new Attribute(attribute)); // Aurretik ateratako atributuak
            }
            attributeList.add(new Attribute("class", (List<String>) null)); // Klase atributua

            // Datu multzoa sortu
            Instances dataDevTest = new Instances("Dataset", attributeList, 0);
            dataDevTest.setClassIndex(attributeList.size() - 1); // Azken atributua klasea da

            // Karpetak zeharkatu eta instantziak gehitu
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        // Fitxategiaren edukia irakurri eta hitzak atera
                        List<String> words = new ArrayList<>();
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                // Hitzak atera (alfanumerikoak soilik)
                                String[] tokens = line.split("\\W+");
                                for (String token : tokens) {
                                    if (!token.isEmpty()) {
                                        words.add(token.toLowerCase()); // Hitzak minuskulaz
                                    }
                                }
                            }
                        } catch (Exception e) {
                            System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                            e.printStackTrace();
                        }

                        // Instantzia sortu eta datuak gehitu
                        DenseInstance instance = new DenseInstance(attributeList.size());
                        for (int i = 0; i < attributes.length; i++) {
                            if (words.contains(attributes[i])) {
                                instance.setValue(attributeList.get(i), 1.0); // Hitzaren presentzia
                            } else {
                                instance.setValue(attributeList.get(i), 0.0); // Hitzaren ez presentzia
                            }
                        }
                        instance.setValue(attributeList.get(attributeList.size() - 1), folder.getName()); // Klasea
                        dataDevTest.add(instance);
                    }
                }
            }
            return dataDevTest;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private void save(Instances attributes, String outFile) {
        try {
            // datuak karpeta eratu
            File dir = new File("datuak");
            if (!dir.exists()) {
                dir.mkdir();
            }
            FileWriter writer = new FileWriter("datuak/" + outFile);
            // .arff fitxategia gorde
            writer.write(attributes.toString());
            writer.close();
            System.out.println("Fitxategia .arff formatuan ongi sortu da hemen: " + outFile);
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void save(String[] data, String outFile) {
        try {
            FileWriter writer = new FileWriter("datuak/" + outFile);
            writer.write(data + "\n");
            writer.close();
            System.out.println("Fitxategia .txt formatuan ongi sortu da hemen: " + outFile);
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // Test
        String inPath = "movies_reviews/";
        String outFile = "datuak";
        datuBilketa.getDB().bildu(inPath, outFile);
    }
}