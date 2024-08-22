package com.lightweight;

import java.io.*;
import java.util.regex.*;

public class RemoveComments {

    // 파일에서 모든 주석을 제거하는 메서드
    public static String removeCommentsFromFile(String inputFilePath) throws IOException {
        // 파일을 읽기 위해 BufferedReader 사용
        BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
        StringBuilder code = new StringBuilder();
        String line;

        // 파일의 모든 라인을 읽어들여 StringBuilder에 추가
        while ((line = reader.readLine()) != null) {
            code.append(line).append("\n");
        }
        reader.close();

        // 주석 제거
        return removeComments(code.toString());
    }

    // 주어진 자바 코드 문자열에서 주석을 제거하는 메서드
    public static String removeComments(String code) {
        String regex = "(?s)/\\*.*?\\*/|//.*?(\r?\n|$)";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(code);
        return matcher.replaceAll("").trim();
    }
}