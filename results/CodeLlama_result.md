# CodeBERT result

### Chart-4
```diff
}
+ if ( r!= null ) {
Collection c = r . getAnnotations ( ) ;
...
}
+ }
}
```
- New lines added
---
### Chart-12
```diff
- this . dataset = dataset ;
+ setDataset ( dataset ) ;
```
---
### Chart-26
```diff
ChartRenderingInfo owner = plotState . getOwner ( ) ;
+ if ( owner!= null ) {
EntityCollection entities = owner . getEntityCollection ( ) ;
...
}
+}
}
```
- New lines added
---
### Closure-18
```diff
- if ( options . dependencyOptions . needsManagement ( ) && options . closurePass ) {
+ if ( options. dependencyOptions. needsManagement ( ) ) {
```
---
### Closure-31
```diff
- if ( options . dependencyOptions . needsManagement ( ) && ! options . skipAllPasses && options . closurePass ) {
+ if ( options. dependencyOptions. needsManagement ( ) && options. closurePass ) {
```
---
### Closure-65
```diff
- sb . append ( \"\\\\0\" ) ;
+ sb. append ( \"\\\\000\" ) ;
```
---
### Closure-70
```diff
- defineSlot ( astParameter , functionNode , jsDocParameter . getJSType ( ) , true ) ;
+ defineSlot ( astParameter, functionNode, jsDocParameter. getJSType ( ), false ) ;
```
---
### Closure-73
```diff
- if ( c > 0x1f && c <= 0x7f ) { 
+ if ( c > 0x1f && c < 0x7f ) {
```
---
### Closure-77
```diff
switch ( c ) {
+ case '\\0': sb. append ( \"\\\\0\" ) ;
+ break ;
case '\\n': sb . append ( \"\\\\n\" ) ;
```
- New lines added
---
### Closure-86
```diff
- return true ;
+ return false ;
```
---
### Closure-92
```diff
- int indexOfDot = namespace . indexOf ( '.' ) ;
+ int indexOfDot = namespace. lastIndexOf ( '.') ;
```
---
### Closure-123
```diff
- Context rhsContext = Context . OTHER ;
+ Context rhsContext = getContextForNoInOperator ( context ) ;
```
---
### Codec-3
```diff
- } else if ( contains ( value , index + 1 , 4 , \"IER\" ) ) {
+ } else if ( contains ( value, index + 1, 3, \"IER\" ) ) {
```
---
### JacksonCore-19
```diff
<method1>
if ( c == '.' ) {
+ if ( outPtr >= outBuf. length ) {
+ outBuf = _textBuffer. finishCurrentSegment ( ) ;
+ outPtr = 0 ;
+ }
outBuf[outPtr ++ ] = c ;

<method2>
if ( c == INT_PERIOD ) {
+ if ( outPtr >= outBuf. length ) {
+ outBuf = _textBuffer. finishCurrentSegment ( ) ;
+ outPtr = 0 ;
+ }
outBuf[outPtr ++ ] = ( char ) c ;
```
- New lines added
---
### JacksonCore-25
```diff
- if ( i <= maxCode ) { 
+ if ( i < maxCode ) { 
```
---
### JacksonDatabind-71
```diff
- if ( raw == String . class || raw == Object . class ) {
+ if ( raw == String. class || raw == Object. class || raw == CharSequence. class ) {
```
---
### JacksonDatabind-83
```diff
- if ( _deserialize ( text , ctxt ) != null ) {
return _deserialize ( text , ctxt ) ;
- }
```
- All lines deleted
---
### Jsoup-35
```diff
Element adopter = new Element ( formatEl . tag ( ) , tb . getBaseUri ( ) ) ; 
+ adopter. attributes ( ). addAll ( formatEl. attributes ( ) ) ;
Node[] childNodes = furthestBlock . childNodes ( ) . toArray ( new Node[furthestBlock . childNodeSize ( ) ] ) ;
```
- New lines added
---
### Lang-16
```diff
- if ( str . startsWith ( \"0x\" ) || str . startsWith ( \"-0x\" ) ) {
+ if ( str. startsWith ( \"0x\" ) || str. startsWith ( \"-0x\" ) || str. startsWith ( \"0X\" ) || str. startsWith ( \"-0X\" ) ) {
```
---
### Lang-24
```diff
- return foundDigit && ! hasExp ;
+ return foundDigit &&! hasExp &&! hasDecPoint ;
```
---
### Lang-39
```diff
for ( int i = 0 ; i < searchList . length ; i ++ ) { 
+ if ( searchList[i] == null || replacementList[i] == null ) {
+ continue ;
+ }
int greater = replacementList[i] . length ( ) - searchList[i] . length ( ) ;
```
- New lines added
---
### Math-32
```diff
- if ( ( Boolean ) tree . getAttribute ( ) ) {
+ if ( ( Boolean ) tree. getAttribute ( ) && tree. getCut ( ) == null ) {
```
---
### Time-4
```diff
- Partial newPartial = new Partial ( iChronology , newTypes , newValues ) ;
+ Partial newPartial = new Partial ( newTypes, newValues ) ;
```
- [MCRepair++](https://github.com/kimjisung78/MCRepair-Plus-Plus/blob/main/results/markdowns/Time_4.md) considered as a Correct Patch.
- The validation result of ```Partial newPartial = new Partial(newTypes, newValues);``` is equivalent to the validation result of ```Partial newPartial = new Partial(newTypes, newValues, iChronology);`