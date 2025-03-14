import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

public class lab6 {
    public static void main(String[] args) {
        String dataSource = args[0];
        String gordeleku = args[1]; // Output

        Instances data = loadData(dataSource);
        if (data == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return;
        }        
    }

    private static Instances loadData(String filePath) { // Oso totxo, igual ez dira behar horrenbeste if
        DataSource source = null;
        try {
            source = new DataSource(filePath);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia aurkitu.");
            return null;
        }
        Instances data = null;
        try {
            data = source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }

        if (data == null) {
            System.out.println("ERROREA: " + filePath + " fitxategiaren edukia hutsik dago.");
            return null;
        }

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        System.out.println("Class index set to: " + data.classIndex());
        return data;
    }
}