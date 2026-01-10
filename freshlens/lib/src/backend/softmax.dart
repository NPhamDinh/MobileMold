import 'dart:math';

List<double> softmax(List<double> input) {
  double maxVal = input.reduce((max, element) => max > element ? max : element);

  List<double> expValues = input.map((val) => exp(val - maxVal)).toList();
  double sumExp = expValues.reduce((sum, element) => sum + element);

  List<double> softmaxValues = expValues.map((val) => val / sumExp).toList();

  return softmaxValues;
}
