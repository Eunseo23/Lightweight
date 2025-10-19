# CodeBERT result

### Chart-1

```diff
- if ( dataset != null ) {
+ if ( dataset == null ) {
```
---
### Chart-7

```diff
- long s = getDataItem ( this . minMiddleIndex ) . getPeriod ( ) . getStart ( ) . getTime ( ) ;
+ long s = getDataItem ( this . maxMiddleIndex ) . getPeriod ( ) . getStart ( ) . getTime ( ) ; 
- long e = getDataItem ( this . minMiddleIndex ) . getPeriod ( ) . getEnd ( ) . getTime ( ) ;
+ long e = getDataItem ( this . maxMiddleIndex ) . getPeriod ( ) . getEnd ( ) . getTime ( ) ;
```
---
### Chart-9
```diff
- if ( endIndex < 0 ) { 
+ if ( endIndex < 0 || endIndex < startIndex ) {
```
---
### Chart-11
```diff
- PathIterator iterator2 = p1 . getPathIterator ( null ) ;
+ PathIterator iterator2 = p2 . getPathIterator ( null ) ;
```
---
### Cli-8
```diff
- pos = findWrapPos ( text , width , nextLineTabStop ) ;
+ pos = findWrapPos ( text , width , 0 ) ;
```
---
### Cli-25
```diff
- nextLineTabStop = width - 1 ; 
+ nextLineTabStop = 1 ;
```
---
### Cli-28
```diff
- break ;
+ continue ;
```
---
### Cli-32
```diff
- while ( ( pos <= text . length ( ) ) && ( ( c = text . charAt ( pos ) ) != ' ' ) && ( c != '\\n' ) && ( c != '\\r' ) ) {
- ++ pos ; 
- } 
```
- All lines deleted
---
### Closure-6
```diff
<method1>
- if ( ( leftType . isConstructor ( ) || leftType . isEnumType ( ) ) && ( rightType . isConstructor ( ) || rightType . isEnumType ( ) ) ) {
- registerMismatch ( rightType , leftType , null ) ; 
- } else {
JSType ownerType = getJSType ( owner ) ;
...
mismatch ( t , n , \"assignment to property \" + propName + \" of \" + getReadableJSTypeName ( owner , true ) , rightType , leftType ) ;
- }

<method2>
- if ( ( leftType . isConstructor ( ) || leftType . isEnumType ( ) ) && ( rightType . isConstructor ( ) || rightType . isEnumType ( ) ) ) { 
- registerMismatch ( rightType , leftType , null ) ;
- } else {
mismatch ( t , n , msg , rightType , leftType ) ;
- }
```
- All lines deleted
---
### Closure-14
```diff
- cfa . createEdge ( fromNode , Branch . UNCOND , finallyNode ) ;
+ cfa . createEdge ( fromNode , Branch . ON_EX , finallyNode ) ;
```
---
### Closure-18
```diff
- if ( options . dependencyOptions . needsManagement ( ) && options . closurePass ) {
+ if ( options . dependencyOptions . needsManagement ( ) ) {
```
---
### Closure-31
```diff
- if ( options . dependencyOptions . needsManagement ( ) && ! options . skipAllPasses && options . closurePass ) {
+ if ( options . dependencyOptions . needsManagement ( ) && options . closurePass ) {
```
---
### Closure-40
```diff
- JsName name = getName ( ns . name , false ) ;
- if ( name != null ) {
refNodes . add ( new ClassDefiningFunctionNode ( name , n , parent , parent . getParent ( ) ) ) ;
- }
```
- All lines deleted
---
### Closure-62
```diff
- if ( excerpt . equals ( LINE ) && 0 <= charno && charno < sourceExcerpt . length ( ) ) {
+ if ( excerpt . equals ( LINE ) && 0 <= charno && charno <= sourceExcerpt . length ( ) ) {
```
---
### Closure-70
```diff
- defineSlot ( astParameter , functionNode , jsDocParameter . getJSType ( ) , true ) ;
+ defineSlot ( astParameter , functionNode , jsDocParameter . getJSType ( ) , false ) ;
```
---
### Closure-73
```diff
- if ( c > 0x1f && c <= 0x7f ) { 
+ if ( c > 0x1f && c < 0x7f ) { 
```
---
### Closure-78
```diff
if ( rval == 0 ) {
- error ( DiagnosticType . error ( \"JSC_DIVIDE_BY_0_ERROR\" , \"Divide by 0\" ) , right ) ;
return null ;
...
if ( rval == 0 ) {
- error ( DiagnosticType . error ( \"JSC_DIVIDE_BY_0_ERROR\" , \"Divide by 0\" ) , right ) ;
return null ;
```
- All lines deleted
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
+ int indexOfDot = namespace . lastIndexOf ( '.' ) ;
```
---
### Closure-115
```diff
- boolean hasSideEffects = false ;
- if ( block . hasChildren ( ) ) {
- Preconditions . checkState ( block . hasOneChild ( ) ) ;
- Node stmt = block . getFirstChild ( ) ;
- if ( stmt . isReturn ( ) ) {
- hasSideEffects = NodeUtil . mayHaveSideEffects ( stmt . getFirstChild ( ) , compiler ) ;
- } 
- }
Node cArg = callNode . getFirstChild ( ) . getNext ( ) ;
...
if ( cArg != null ) {
- if ( hasSideEffects && NodeUtil . canBeSideEffected ( cArg ) ) {
- return CanInlineResult . NO ;
- }
```
- All lines deleted
---
### Closure-126
```diff
- if ( NodeUtil . hasFinally ( n ) ) {
- Node finallyBlock = n . getLastChild ( ) ;
- tryMinimizeExits ( finallyBlock , exitType , labelName ) ;
- }
```
- All lines deleted
---
### Closure-168
```diff
- if ( t . getScopeDepth ( ) <= 2 ) { 
+ if ( t . getScopeDepth ( ) <= 1 ) {
```
---
### Codec-2
```diff
- if ( lineLength > 0 ) {
+ if ( lineLength > 0 && pos > 0 ) {
```
---
### Codec-3
```diff
- } else if ( contains ( value , index + 1 , 4 , \"IER\" ) ) {
+ } else if ( contains ( value , index + 1 , 3 , \"IER\" ) ) {
```
---
### Compress-32
```diff
- currEntry . setGroupId ( Integer . parseInt ( val ) ) ;
+ currEntry . setGroupId ( Long . parseLong ( val ) ) ;
...
- currEntry . setUserId ( Integer . parseInt ( val ) ) ;
+ currEntry . setUserId ( Long . parseLong ( val ) ) ;
```
---
### Csv-11
```diff
- final boolean emptyHeader = header . trim ( ) . isEmpty ( ) ;  
+ final boolean emptyHeader = ( header == null || header . trim ( ) . isEmpty ( ) ) ;
```
---
### JacksonCore-25
```diff
- if ( i <= maxCode ) { 
+ if ( i < maxCode ) { 
```
---
### JacksonDatabind-1
```diff
} 
+ return ;
}
```
- New lines added
---
### JacksonDatabind-17
```diff
- return ( t . getRawClass ( ) == Object . class ) || ( ! t . isConcrete ( ) || TreeNode . class . isAssignableFrom ( t . getRawClass ( ) ) ) ;
+ return ( t . getRawClass ( ) == Object . class ) || ( ! t . isConcrete ( ) && ! TreeNode . class . isAssignableFrom ( t . getRawClass ( ) ) ) ;
```
---
### JacksonDatabind-27
```diff
- if ( ext . handlePropertyValue ( p , ctxt , propName , buffer ) ) { ;
+ if ( ext . handlePropertyValue ( p , ctxt , propName , null ) ) { ;
```
---
### JacksonDatabind-67
```diff
- return _createEnumKeyDeserializer ( ctxt , type ) ;
+ deser = _createEnumKeyDeserializer ( ctxt , type ) ;
- }
+ } else {
deser = StdKeyDeserializers . findStringBasedKeyDeserializer ( config , type ) ;
+ }
}
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
### JacksonDatabind-96
```diff
- paramName = candidate . findImplicitParamName ( 0 ) ;
+ paramName = candidate . paramName ( 0 ) ;
```
---
### JacksonDatabind-102
```diff
- if ( property == null ) {
- return this ;
- }
```
- All lines deleted
---
### Jsoup-39
```diff
charsetName = defaultCharset ;
+ doc = null ;
if ( doc == null ) {
```
- New lines added
---
### JxPath-5
```diff
- throw new JXPathException ( \"Cannot compare pointers that do not belong to the same tree : '\" + p1 + \"' and '\" + p2 + \"'\" ) ;
+ return 0 ;
```
---
### Lang-10
```diff
- boolean wasWhite = false ;
for ( int i = 0 ; i < value . length ( ) ; ++ i ) {
char c = value . charAt ( i ) ; 
- if ( Character . isWhitespace ( c ) ) {
- if ( ! wasWhite ) {
- wasWhite = true ;
- regex . append ( \"\\\\s * +\" ) ;
- }
- continue ;
- }
- wasWhite = false ;
```
- All lines deleted
---
### Lang-24
```diff
- return foundDigit && ! hasExp ;
+ return foundDigit && ! hasExp && ! hasDecPoint ;
```
---
### Lang-43
```diff
if ( escapingOn && c[start] == QUOTE ) {
+ next ( pos ) ;
return appendTo == null ? null : appendTo . append ( QUOTE ) ; 
```
- New lines added
---
### Lang-51
```diff
}
+ return false ;
}
```
- New lines added
---
### Math-32
```diff
- if ( ( Boolean ) tree . getAttribute ( ) ) {
+ if ( tree . getAttribute ( ) instanceof Boolean ) {
```
- [MCRepair++](https://github.com/kimjisung78/MCRepair-Plus-Plus/blob/main/results/markdowns/Math_32.md) considered as a Correct Patch.
- To fix the bug, its source code must be changed that it checks that ```tree.getAttribute()``` can convert to ```Boolean``` class.
---
### Math-50
```diff
- if ( x == x1 ) {
- x0 = 0 . 5 * ( x0 + x1 - FastMath . max ( rtol * FastMath . abs ( x1 ) , atol ) ) ;
- f0 = computeObjectiveValue ( x0 ) ;
- } 
```
- All lines deleted
---
### Math-57
```diff
- int sum = 0 ;
+ double sum = 0 ;
```
---
### Time-4
```diff
- Partial newPartial = new Partial ( iChronology , newTypes , newValues ) ;
+ Partial newPartial = new Partial ( newTypes , newValues ) ;
```
- [MCRepair++](https://github.com/kimjisung78/MCRepair-Plus-Plus/blob/main/results/markdowns/Time_4.md) considered as a Correct Patch.
- The validation result of ```Partial newPartial = new Partial(newTypes, newValues);``` is equivalent to the validation result of ```Partial newPartial = new Partial(newTypes, newValues, iChronology);```
