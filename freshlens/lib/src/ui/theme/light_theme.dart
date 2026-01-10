import 'package:flutter/material.dart';

import 'constants.dart';

final lightTheme = ThemeData(
  primaryColor: black,
  colorScheme: colorScheme,
  fontFamily: "Nunito",
  progressIndicatorTheme:
      const ProgressIndicatorThemeData(color: black, linearMinHeight: 2),
  textTheme: textTheme,
  elevatedButtonTheme: elevatedButtonTheme,
  outlinedButtonTheme: outlinedButtonTheme,
  popupMenuTheme: popupMenuTheme,
  scaffoldBackgroundColor: white,
  canvasColor: white,
  iconTheme: const IconThemeData(color: black, size: 20),
  cardTheme: cardTheme,
  inputDecorationTheme: inputDecorationTheme,
  dividerTheme: dividerTheme,
  textSelectionTheme: textSelectionTheme,
  snackBarTheme: SnackBarThemeData(
    shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kBorderRadiusMultiplier)),
    elevation: kElevationMultiplier,
    behavior: SnackBarBehavior.floating,
    backgroundColor: secondary,
  ),
);

const ColorScheme colorScheme = ColorScheme(
  brightness: Brightness.light,
  surface: white,
  onSurface: black,
  surfaceContainerHighest: black,
  primary: black,
  onPrimary: white,
  primaryContainer: accent,
  onPrimaryContainer: white,
  secondary: accent,
  onSecondary: white,
  error: secondary,
  onError: white,
  tertiary: secondary,
  secondaryContainer: white,
  onSecondaryContainer: black,
);

const TextSelectionThemeData textSelectionTheme =
    TextSelectionThemeData(selectionColor: Color.fromARGB(149, 0, 96, 251));

InputDecorationTheme inputDecorationTheme = InputDecorationTheme(
    floatingLabelBehavior: FloatingLabelBehavior.always,
    floatingLabelAlignment: FloatingLabelAlignment.start,
    focusedBorder: OutlineInputBorder(
        borderSide: const BorderSide(color: black, width: kBorderWidth),
        borderRadius: BorderRadius.circular(kBorderRadiusMultiplier)),
    border: OutlineInputBorder(
      borderSide: const BorderSide(color: black, width: kBorderWidth),
      borderRadius: BorderRadius.circular(kBorderRadiusMultiplier),
    ),
    enabledBorder: OutlineInputBorder(
      borderSide: const BorderSide(color: black, width: kBorderWidth),
      borderRadius: BorderRadius.circular(kBorderRadiusMultiplier),
    ),
    errorBorder: OutlineInputBorder(
      borderSide: const BorderSide(color: secondary, width: kBorderWidth),
      borderRadius: BorderRadius.circular(kBorderRadiusMultiplier),
    ),
    focusedErrorBorder: OutlineInputBorder(
      borderSide: const BorderSide(color: secondary, width: 2 * kBorderWidth),
      borderRadius: BorderRadius.circular(kBorderRadiusMultiplier),
    ),
    contentPadding: const EdgeInsets.symmetric(
        horizontal: kDefaultPadding / 2, vertical: kDefaultPadding));

final cardTheme = CardThemeData(
    clipBehavior: Clip.antiAliasWithSaveLayer,
    elevation: kElevationMultiplier * kSmallElevationMultiplier,
    margin: EdgeInsets.zero,
    shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(1.5 * kBorderRadiusMultiplier),
        side: const BorderSide(color: grey, width: kBorderWidth)));

const textTheme = TextTheme(
    bodyLarge: TextStyle(color: black),
    bodySmall: TextStyle(color: black),
    bodyMedium: TextStyle(color: black, fontSize: 13),
    headlineLarge: TextStyle(
        color: black,
        fontWeight: FontWeight.w900,
        fontSize: 35,
        fontFamily: "MPLUSRounded"),
    headlineMedium: TextStyle(
        color: black,
        fontWeight: FontWeight.w900,
        fontSize: 28,
        fontFamily: "MPLUSRounded"),
    displaySmall: TextStyle(
        fontFamily: "MPLUSRounded", fontWeight: FontWeight.w900, fontSize: 12));

final elevatedButtonTheme = ElevatedButtonThemeData(
  style: ButtonStyle(
      elevation:
          WidgetStateProperty.resolveWith<double>((Set<WidgetState> states) {
        if (states.contains(WidgetState.hovered)) {
          return 1.5 * kElevationMultiplier;
        }
        return kElevationMultiplier;
      }),
      shape: WidgetStateProperty.all(RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(kBorderRadiusMultiplier))),
      padding: WidgetStateProperty.all(const EdgeInsets.all(kButtonPadding)),
      backgroundColor: WidgetStateProperty.all(accent),
      minimumSize: WidgetStateProperty.all(const Size(250, 0))),
);

final outlinedButtonTheme = OutlinedButtonThemeData(
  style: ButtonStyle(
    textStyle: WidgetStateProperty.all(textTheme.displaySmall),
    elevation:
        WidgetStateProperty.resolveWith<double>((Set<WidgetState> states) {
      return states.contains(WidgetState.hovered) ? 10 : 5;
    }),
    shape: WidgetStateProperty.all(RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kBorderRadiusMultiplier))),
    padding: WidgetStateProperty.all(const EdgeInsets.symmetric(
        horizontal: kButtonPadding, vertical: 1.2 * kButtonPadding)),
    backgroundColor: WidgetStateProperty.all(white),
    foregroundColor: WidgetStateProperty.all(black),
    side: WidgetStateProperty.resolveWith((Set<WidgetState> states) {
      if (states.contains(WidgetState.hovered) ||
          states.contains(WidgetState.focused)) {
        return const BorderSide(color: black);
      }
      return const BorderSide(color: black, width: kBorderWidth);
    }),
    //overlayColor: WidgetStateProperty.all(Colors.transparent),
  ),
);

final popupMenuTheme = PopupMenuThemeData(
    textStyle: textTheme.bodyMedium,
    elevation: kSmallElevationMultiplier * kElevationMultiplier,
    shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(kBorderRadiusMultiplier)));

const DividerThemeData dividerTheme = DividerThemeData(color: grey);
