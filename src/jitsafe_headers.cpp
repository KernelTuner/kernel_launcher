#include <string>
#include <unordered_map>

namespace kernel_launcher {

// These headers files were take from jitify: https://github.com/NVIDIA/jitify/
// They are stripped down headers that are compatible with NVRTC

static const char* jitsafe_header_float_h = R"(
#ifndef __CFLOAT_HEADER
#define __CFLOAT_HEADER
#define FLT_RADIX       2
#define FLT_MANT_DIG    24
#define DBL_MANT_DIG    53
#define FLT_DIG         6
#define DBL_DIG         15
#define FLT_MIN_EXP     -125
#define DBL_MIN_EXP     -1021
#define FLT_MIN_10_EXP  -37
#define DBL_MIN_10_EXP  -307
#define FLT_MAX_EXP     128
#define DBL_MAX_EXP     1024
#define FLT_MAX_10_EXP  38
#define DBL_MAX_10_EXP  308
#define FLT_MAX         3.4028234e38f
#define DBL_MAX         1.7976931348623157e308
#define FLT_EPSILON     1.19209289e-7f
#define DBL_EPSILON     2.220440492503130e-16
#define FLT_MIN         1.1754943e-38f
#define DBL_MIN         2.2250738585072013e-308
#define FLT_ROUNDS      1
#if defined __cplusplus && __cplusplus >= 201103L
#define FLT_EVAL_METHOD 0
#define DECIMAL_DIG     21
#endif
#endif
)";

static const char* jitsafe_header_limits_h = R"(
#ifndef __LIMITS_HEADER
#define __LIMITS_HEADER
#if defined _WIN32 || defined _WIN64
 #define __WORDSIZE 32
#else
 #if defined(__LP64__) || (defined __x86_64__ && !defined __ILP32__)
  #define __WORDSIZE 64
 #else
  #define __WORDSIZE 32
 #endif
#endif
#define MB_LEN_MAX  16
#define CHAR_BIT    8
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127
#define UCHAR_MAX   255
enum {
  _JITIFY_CHAR_IS_UNSIGNED = (char)-1 >= 0,
  CHAR_MIN = _JITIFY_CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN,
  CHAR_MAX = _JITIFY_CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX,
};
#define SHRT_MIN    (-SHRT_MAX - 1)
#define SHRT_MAX    0x7fff
#define USHRT_MAX   0xffff
#define INT_MIN     (-INT_MAX - 1)
#define INT_MAX     0x7fffffff
#define UINT_MAX    0xffffffff
#if __WORDSIZE == 64
 # define LONG_MAX  LLONG_MAX
#else
 # define LONG_MAX  UINT_MAX
#endif
#define LONG_MIN    (-LONG_MAX - 1)
#if __WORDSIZE == 64
 #define ULONG_MAX  ULLONG_MAX
#else
 #define ULONG_MAX  UINT_MAX
#endif
#define LLONG_MAX  0x7fffffffffffffff
#define LLONG_MIN  (-LLONG_MAX - 1)
#define ULLONG_MAX 0xffffffffffffffff
#endif
)";

static const char* jitsafe_header_iterator = R"(
#ifndef __ITERATOR_HEADER
#define __ITERATOR_HEADER
namespace std {
struct output_iterator_tag {};
struct input_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};
template<class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};
template<class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};
template<class T>
struct iterator_traits<T const*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T const*                   pointer;
  typedef T const&                   reference;
};
}  // namespace std
#endif
)";

// TODO: This is incomplete; need floating point limits
//   Joe Eaton: added IEEE float and double types, none of the smaller types
//              using type specific structs since we can't template on floats.
static const char* jitsafe_header_limits = R"(
#ifndef __LIMITS_HEADER
#define __LIMITS_HEADER
#include <cfloat>
#include <climits>
#include <cstdint>
// TODO: epsilon(), infinity(), etc
namespace std {
namespace __jitify_detail {
#if __cplusplus >= 201103L
#define JITIFY_CXX11_CONSTEXPR constexpr
#define JITIFY_CXX11_NOEXCEPT noexcept
#else
#define JITIFY_CXX11_CONSTEXPR
#define JITIFY_CXX11_NOEXCEPT
#endif
struct FloatLimits {
#if __cplusplus >= 201103L
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          float lowest() JITIFY_CXX11_NOEXCEPT {   return -FLT_MAX;}
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          float min() JITIFY_CXX11_NOEXCEPT {      return FLT_MIN; }
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          float max() JITIFY_CXX11_NOEXCEPT {      return FLT_MAX; }
#endif  // __cplusplus >= 201103L
   enum {
   is_specialized    = true,
   is_signed         = true,
   is_integer        = false,
   is_exact          = false,
   has_infinity      = true,
   has_quiet_NaN     = true,
   has_signaling_NaN = true,
   has_denorm        = 1,
   has_denorm_loss   = true,
   round_style       = 1,
   is_iec559         = true,
   is_bounded        = true,
   is_modulo         = false,
   digits            = 24,
   digits10          = 6,
   max_digits10      = 9,
   radix             = 2,
   min_exponent      = -125,
   min_exponent10    = -37,
   max_exponent      = 128,
   max_exponent10    = 38,
   tinyness_before   = false,
   traps             = false
   };
};
struct DoubleLimits {
#if __cplusplus >= 201103L
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          double lowest() noexcept { return -DBL_MAX; }
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          double min() noexcept { return DBL_MIN; }
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          double max() noexcept { return DBL_MAX; }
#endif  // __cplusplus >= 201103L
   enum {
   is_specialized    = true,
   is_signed         = true,
   is_integer        = false,
   is_exact          = false,
   has_infinity      = true,
   has_quiet_NaN     = true,
   has_signaling_NaN = true,
   has_denorm        = 1,
   has_denorm_loss   = true,
   round_style       = 1,
   is_iec559         = true,
   is_bounded        = true,
   is_modulo         = false,
   digits            = 53,
   digits10          = 15,
   max_digits10      = 17,
   radix             = 2,
   min_exponent      = -1021,
   min_exponent10    = -307,
   max_exponent      = 1024,
   max_exponent10    = 308,
   tinyness_before   = false,
   traps             = false
   };
};
template<class T, T Min, T Max, int Digits=-1>
struct IntegerLimits {
	static inline __host__ __device__ T min() { return Min; }
	static inline __host__ __device__ T max() { return Max; }
#if __cplusplus >= 201103L
	static constexpr inline __host__ __device__ T lowest() noexcept {
		return Min;
	}
#endif  // __cplusplus >= 201103L
	enum {
       is_specialized = true,
       digits            = (Digits == -1) ? (int)(sizeof(T)*8 - (Min != 0)) : Digits,
       digits10          = (digits * 30103) / 100000,
       is_signed         = ((T)(-1)<0),
       is_integer        = true,
       is_exact          = true,
       has_infinity      = false,
       has_quiet_NaN     = false,
       has_signaling_NaN = false,
       has_denorm        = 0,
       has_denorm_loss   = false,
       round_style       = 0,
       is_iec559         = false,
       is_bounded        = true,
       is_modulo         = !(is_signed || Max == 1 /*is bool*/),
       max_digits10      = 0,
       radix             = 2,
       min_exponent      = 0,
       min_exponent10    = 0,
       max_exponent      = 0,
       max_exponent10    = 0,
       tinyness_before   = false,
       traps             = false
	};
};
} // namespace __jitify_detail
template<typename T> struct numeric_limits {
    enum { is_specialized = false };
};
template<> struct numeric_limits<bool>               : public 
__jitify_detail::IntegerLimits<bool,              false,    true,1> {};
template<> struct numeric_limits<char>               : public 
__jitify_detail::IntegerLimits<char,              CHAR_MIN, CHAR_MAX> 
{};
template<> struct numeric_limits<signed char>        : public 
__jitify_detail::IntegerLimits<signed char,       SCHAR_MIN,SCHAR_MAX> 
{};
template<> struct numeric_limits<unsigned char>      : public 
__jitify_detail::IntegerLimits<unsigned char,     0,        UCHAR_MAX> 
{};
template<> struct numeric_limits<wchar_t>            : public 
__jitify_detail::IntegerLimits<wchar_t,           WCHAR_MIN, WCHAR_MAX> {};
template<> struct numeric_limits<short>              : public 
__jitify_detail::IntegerLimits<short,             SHRT_MIN, SHRT_MAX> 
{};
template<> struct numeric_limits<unsigned short>     : public 
__jitify_detail::IntegerLimits<unsigned short,    0,        USHRT_MAX> 
{};
template<> struct numeric_limits<int>                : public 
__jitify_detail::IntegerLimits<int,               INT_MIN,  INT_MAX> {};
template<> struct numeric_limits<unsigned int>       : public 
__jitify_detail::IntegerLimits<unsigned int,      0,        UINT_MAX> 
{};
template<> struct numeric_limits<long>               : public 
__jitify_detail::IntegerLimits<long,              LONG_MIN, LONG_MAX> 
{};
template<> struct numeric_limits<unsigned long>      : public 
__jitify_detail::IntegerLimits<unsigned long,     0,        ULONG_MAX> 
{};
template<> struct numeric_limits<long long>          : public 
__jitify_detail::IntegerLimits<long long,         LLONG_MIN,LLONG_MAX> 
{};
template<> struct numeric_limits<unsigned long long> : public 
__jitify_detail::IntegerLimits<unsigned long long,0,        ULLONG_MAX> 
{};
//template<typename T> struct numeric_limits { static const bool 
//is_signed = ((T)(-1)<0); };
template<> struct numeric_limits<float>              : public 
__jitify_detail::FloatLimits 
{};
template<> struct numeric_limits<double>             : public 
__jitify_detail::DoubleLimits 
{};
}  // namespace std
#endif
)";

// TODO: This is highly incomplete
static const char* jitsafe_header_type_traits = R"(
#ifndef __TYPE_TRAITS_HEADER
#define __TYPE_TRAITS_HEADER
#if __cplusplus >= 201103L
namespace std {
template<bool B, class T = void> struct enable_if {};
template<class T>                struct enable_if<true, T> { typedef T type; };
#if __cplusplus >= 201402L
template< bool B, class T = void > using enable_if_t = typename enable_if<B,T>::type;
#endif
struct true_type  {
  enum { value = true };
  operator bool() const { return true; }
};
struct false_type {
  enum { value = false };
  operator bool() const { return false; }
};
template<typename T> struct is_floating_point    : false_type {};
template<> struct is_floating_point<float>       :  true_type {};
template<> struct is_floating_point<double>      :  true_type {};
template<> struct is_floating_point<long double> :  true_type {};
#if __cplusplus >= 201703L
template<typename T> inline constexpr bool is_floating_point_v = is_floating_point<T>::value;
#endif  // __cplusplus >= 201703L
template<class T> struct is_integral              : false_type {};
template<> struct is_integral<bool>               :  true_type {};
template<> struct is_integral<char>               :  true_type {};
template<> struct is_integral<signed char>        :  true_type {};
template<> struct is_integral<unsigned char>      :  true_type {};
template<> struct is_integral<short>              :  true_type {};
template<> struct is_integral<unsigned short>     :  true_type {};
template<> struct is_integral<int>                :  true_type {};
template<> struct is_integral<unsigned int>       :  true_type {};
template<> struct is_integral<long>               :  true_type {};
template<> struct is_integral<unsigned long>      :  true_type {};
template<> struct is_integral<long long>          :  true_type {};
template<> struct is_integral<unsigned long long> :  true_type {};
#if __cplusplus >= 201703L
template<typename T> inline constexpr bool is_integral_v = is_integral<T>::value;
#endif  // __cplusplus >= 201703L
template<typename T> struct is_signed    : false_type {};
template<> struct is_signed<float>       :  true_type {};
template<> struct is_signed<double>      :  true_type {};
template<> struct is_signed<long double> :  true_type {};
template<> struct is_signed<signed char> :  true_type {};
template<> struct is_signed<short>       :  true_type {};
template<> struct is_signed<int>         :  true_type {};
template<> struct is_signed<long>        :  true_type {};
template<> struct is_signed<long long>   :  true_type {};
template<typename T> struct is_unsigned             : false_type {};
template<> struct is_unsigned<unsigned char>      :  true_type {};
template<> struct is_unsigned<unsigned short>     :  true_type {};
template<> struct is_unsigned<unsigned int>       :  true_type {};
template<> struct is_unsigned<unsigned long>      :  true_type {};
template<> struct is_unsigned<unsigned long long> :  true_type {};
template<typename T, typename U> struct is_same      : false_type {};
template<typename T>             struct is_same<T,T> :  true_type {};
#if __cplusplus >= 201703L
template<typename T, typename U> inline constexpr bool is_same_v = is_same<T, U>::value;
#endif  // __cplusplus >= 201703L
template<class T> struct is_array : false_type {};
template<class T> struct is_array<T[]> : true_type {};
template<class T, size_t N> struct is_array<T[N]> : true_type {};
//partial implementation only of is_function
template<class> struct is_function : false_type { };
template<class Ret, class... Args> struct is_function<Ret(Args...)> : true_type {}; //regular
template<class Ret, class... Args> struct is_function<Ret(Args......)> : true_type {}; // variadic
template<class> struct result_of;
template<class F, typename... Args>
struct result_of<F(Args...)> {
// TODO: This is a hack; a proper implem is quite complicated.
typedef typename F::result_type type;
};
template<class T> struct is_pointer                    : false_type {};
template<class T> struct is_pointer<T*>                : true_type {};
template<class T> struct is_pointer<T* const>          : true_type {};
template<class T> struct is_pointer<T* volatile>       : true_type {};
template<class T> struct is_pointer<T* const volatile> : true_type {};
#if __cplusplus >= 201703L
template< class T > inline constexpr bool is_pointer_v = is_pointer<T>::value;
#endif  // __cplusplus >= 201703L
template <class T> struct remove_pointer { typedef T type; };
template <class T> struct remove_pointer<T*> { typedef T type; };
template <class T> struct remove_pointer<T* const> { typedef T type; };
template <class T> struct remove_pointer<T* volatile> { typedef T type; };
template <class T> struct remove_pointer<T* const volatile> { typedef T type; };
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };
#if __cplusplus >= 201402L
template< class T > using remove_reference_t = typename remove_reference<T>::type;
#endif
template<class T> struct remove_extent { typedef T type; };
template<class T> struct remove_extent<T[]> { typedef T type; };
template<class T, size_t N> struct remove_extent<T[N]> { typedef T type; };
#if __cplusplus >= 201402L
template< class T > using remove_extent_t = typename remove_extent<T>::type;
#endif
template< class T > struct remove_const          { typedef T type; };
template< class T > struct remove_const<const T> { typedef T type; };
template< class T > struct remove_volatile             { typedef T type; };
template< class T > struct remove_volatile<volatile T> { typedef T type; };
template< class T > struct remove_cv { typedef typename remove_volatile<typename remove_const<T>::type>::type type; };
#if __cplusplus >= 201402L
template< class T > using remove_cv_t       = typename remove_cv<T>::type;
template< class T > using remove_const_t    = typename remove_const<T>::type;
template< class T > using remove_volatile_t = typename remove_volatile<T>::type;
#endif
template<bool B, class T, class F> struct conditional { typedef T type; };
template<class T, class F> struct conditional<false, T, F> { typedef F type; };
#if __cplusplus >= 201402L
template< bool B, class T, class F > using conditional_t = typename conditional<B,T,F>::type;
#endif
namespace __jitify_detail {
template< class T, bool is_function_type = false > struct add_pointer { using type = typename remove_reference<T>::type*; };
template< class T > struct add_pointer<T, true> { using type = T; };
template< class T, class... Args > struct add_pointer<T(Args...), true> { using type = T(*)(Args...); };
template< class T, class... Args > struct add_pointer<T(Args..., ...), true> { using type = T(*)(Args..., ...); };
}  // namespace __jitify_detail
template< class T > struct add_pointer : __jitify_detail::add_pointer<T, is_function<T>::value> {};
#if __cplusplus >= 201402L
template< class T > using add_pointer_t = typename add_pointer<T>::type;
#endif
template< class T > struct decay {
private:
  typedef typename remove_reference<T>::type U;
public:
  typedef typename conditional<is_array<U>::value, typename remove_extent<U>::type*,
    typename conditional<is_function<U>::value,typename add_pointer<U>::type,typename remove_cv<U>::type
    >::type>::type type;
};
#if __cplusplus >= 201402L
template< class T > using decay_t = typename decay<T>::type;
#endif
template<class T, T v>
struct integral_constant {
static constexpr T value = v;
typedef T value_type;
typedef integral_constant type; // using injected-class-name
constexpr operator value_type() const noexcept { return value; }
#if __cplusplus >= 201402L
constexpr value_type operator()() const noexcept { return value; }
#endif
};
template<typename T> struct is_arithmetic :
std::integral_constant<bool, std::is_integral<T>::value ||
                             std::is_floating_point<T>::value> {};
#if __cplusplus >= 201703L
template<typename T> inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;
#endif  // __cplusplus >= 201703L
template<class T> struct is_lvalue_reference : false_type {};
template<class T> struct is_lvalue_reference<T&> : true_type {};
template<class T> struct is_rvalue_reference : false_type {};
template<class T> struct is_rvalue_reference<T&&> : true_type {};
namespace __jitify_detail {
template <class T> struct type_identity { using type = T; };
template <class T> auto add_lvalue_reference(int) -> type_identity<T&>;
template <class T> auto add_lvalue_reference(...) -> type_identity<T>;
template <class T> auto add_rvalue_reference(int) -> type_identity<T&&>;
template <class T> auto add_rvalue_reference(...) -> type_identity<T>;
} // namespace _jitify_detail
template <class T> struct add_lvalue_reference : decltype(__jitify_detail::add_lvalue_reference<T>(0)) {};
template <class T> struct add_rvalue_reference : decltype(__jitify_detail::add_rvalue_reference<T>(0)) {};
#if __cplusplus >= 201402L
template <class T> using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;
template <class T> using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;
#endif
template<typename T> struct is_const          : public false_type {};
template<typename T> struct is_const<const T> : public true_type {};
template<typename T> struct is_volatile             : public false_type {};
template<typename T> struct is_volatile<volatile T> : public true_type {};
template<typename T> struct is_void             : public false_type {};
template<>           struct is_void<void>       : public true_type {};
template<>           struct is_void<const void> : public true_type {};
template<typename T> struct is_reference     : public false_type {};
template<typename T> struct is_reference<T&> : public true_type {};
template<typename _Tp, bool = (is_void<_Tp>::value || is_reference<_Tp>::value)>
struct __add_reference_helper { typedef _Tp&    type; };
template<typename _Tp> struct __add_reference_helper<_Tp, true> { typedef _Tp     type; };
template<typename _Tp> struct add_reference : public __add_reference_helper<_Tp>{};
namespace __jitify_detail {
template<typename T> struct is_int_or_cref {
typedef typename remove_reference<T>::type type_sans_ref;
static const bool value = (is_integral<T>::value || (is_integral<type_sans_ref>::value
  && is_const<type_sans_ref>::value && !is_volatile<type_sans_ref>::value));
}; // end is_int_or_cref
template<typename From, typename To> struct is_convertible_sfinae {
private:
typedef char                          yes;
typedef struct { char two_chars[2]; } no;
static inline yes   test(To) { return yes(); }
static inline no    test(...) { return no(); }
static inline typename remove_reference<From>::type& from() { typename remove_reference<From>::type* ptr = 0; return *ptr; }
public:
static const bool value = sizeof(test(from())) == sizeof(yes);
}; // end is_convertible_sfinae
template<typename From, typename To> struct is_convertible_needs_simple_test {
static const bool from_is_void      = is_void<From>::value;
static const bool to_is_void        = is_void<To>::value;
static const bool from_is_float     = is_floating_point<typename remove_reference<From>::type>::value;
static const bool to_is_int_or_cref = is_int_or_cref<To>::value;
static const bool value = (from_is_void || to_is_void || (from_is_float && to_is_int_or_cref));
}; // end is_convertible_needs_simple_test
template<typename From, typename To, bool = is_convertible_needs_simple_test<From,To>::value>
struct is_convertible {
static const bool value = (is_void<To>::value || (is_int_or_cref<To>::value && !is_void<From>::value));
}; // end is_convertible
template<typename From, typename To> struct is_convertible<From, To, false> {
static const bool value = (is_convertible_sfinae<typename add_reference<From>::type, To>::value);
}; // end is_convertible
} // end __jitify_detail
// implementation of is_convertible taken from thrust's pre C++11 path
template<typename From, typename To> struct is_convertible
: public integral_constant<bool, __jitify_detail::is_convertible<From, To>::value>
{ }; // end is_convertible
template<class A, class B> struct is_base_of { };
template<size_t len, size_t alignment> struct aligned_storage { struct type { alignas(alignment) char data[len]; }; };
template <class T> struct alignment_of : std::integral_constant<size_t,alignof(T)> {};
template <typename T> struct make_unsigned;
template <> struct make_unsigned<signed char>        { typedef unsigned char type; };
template <> struct make_unsigned<signed short>       { typedef unsigned short type; };
template <> struct make_unsigned<signed int>         { typedef unsigned int type; };
template <> struct make_unsigned<signed long>        { typedef unsigned long type; };
template <> struct make_unsigned<signed long long>   { typedef unsigned long long type; };
template <> struct make_unsigned<unsigned char>      { typedef unsigned char type; };
template <> struct make_unsigned<unsigned short>     { typedef unsigned short type; };
template <> struct make_unsigned<unsigned int>       { typedef unsigned int type; };
template <> struct make_unsigned<unsigned long>      { typedef unsigned long type; };
template <> struct make_unsigned<unsigned long long> { typedef unsigned long long type; };
template <> struct make_unsigned<char>               { typedef unsigned char type; };
#if defined _WIN32 || defined _WIN64
template <> struct make_unsigned<wchar_t>            { typedef unsigned short type; };
#else
template <> struct make_unsigned<wchar_t>            { typedef unsigned int type; };
#endif
template <typename T> struct make_signed;
template <> struct make_signed<signed char>        { typedef signed char type; };
template <> struct make_signed<signed short>       { typedef signed short type; };
template <> struct make_signed<signed int>         { typedef signed int type; };
template <> struct make_signed<signed long>        { typedef signed long type; };
template <> struct make_signed<signed long long>   { typedef signed long long type; };
template <> struct make_signed<unsigned char>      { typedef signed char type; };
template <> struct make_signed<unsigned short>     { typedef signed short type; };
template <> struct make_signed<unsigned int>       { typedef signed int type; };
template <> struct make_signed<unsigned long>      { typedef signed long type; };
template <> struct make_signed<unsigned long long> { typedef signed long long type; };
template <> struct make_signed<char>               { typedef signed char type; };
#if defined _WIN32 || defined _WIN64
template <> struct make_signed<wchar_t>            { typedef signed short type; };
#else
template <> struct make_signed<wchar_t>            { typedef signed int type; };
#endif
}  // namespace std
#endif // c++11
#endif
)";

// TODO: INT_FAST8_MAX et al. and a few other misc constants
static const char* jitsafe_header_stdint_h =
    "#ifndef __STDINT_HEADER\n"
    "#define __STDINT_HEADER\n"
    "#include <climits>\n"
    "namespace __jitify_stdint_ns {\n"
    "typedef signed char      int8_t;\n"
    "typedef signed short     int16_t;\n"
    "typedef signed int       int32_t;\n"
    "typedef signed long long int64_t;\n"
    "typedef signed char      int_fast8_t;\n"
    "typedef signed short     int_fast16_t;\n"
    "typedef signed int       int_fast32_t;\n"
    "typedef signed long long int_fast64_t;\n"
    "typedef signed char      int_least8_t;\n"
    "typedef signed short     int_least16_t;\n"
    "typedef signed int       int_least32_t;\n"
    "typedef signed long long int_least64_t;\n"
    "typedef signed long long intmax_t;\n"
    "typedef signed long      intptr_t; //optional\n"
    "typedef unsigned char      uint8_t;\n"
    "typedef unsigned short     uint16_t;\n"
    "typedef unsigned int       uint32_t;\n"
    "typedef unsigned long long uint64_t;\n"
    "typedef unsigned char      uint_fast8_t;\n"
    "typedef unsigned short     uint_fast16_t;\n"
    "typedef unsigned int       uint_fast32_t;\n"
    "typedef unsigned long long uint_fast64_t;\n"
    "typedef unsigned char      uint_least8_t;\n"
    "typedef unsigned short     uint_least16_t;\n"
    "typedef unsigned int       uint_least32_t;\n"
    "typedef unsigned long long uint_least64_t;\n"
    "typedef unsigned long long uintmax_t;\n"
    "#define INT8_MIN    SCHAR_MIN\n"
    "#define INT16_MIN   SHRT_MIN\n"
    "#if defined _WIN32 || defined _WIN64\n"
    "#define WCHAR_MIN   0\n"
    "#define WCHAR_MAX   USHRT_MAX\n"
    "typedef unsigned long long uintptr_t; //optional\n"
    "#else\n"
    "#define WCHAR_MIN   INT_MIN\n"
    "#define WCHAR_MAX   INT_MAX\n"
    "typedef unsigned long      uintptr_t; //optional\n"
    "#endif\n"
    "#define INT32_MIN   INT_MIN\n"
    "#define INT64_MIN   LLONG_MIN\n"
    "#define INT8_MAX    SCHAR_MAX\n"
    "#define INT16_MAX   SHRT_MAX\n"
    "#define INT32_MAX   INT_MAX\n"
    "#define INT64_MAX   LLONG_MAX\n"
    "#define UINT8_MAX   UCHAR_MAX\n"
    "#define UINT16_MAX  USHRT_MAX\n"
    "#define UINT32_MAX  UINT_MAX\n"
    "#define UINT64_MAX  ULLONG_MAX\n"
    "#define INTPTR_MIN  LONG_MIN\n"
    "#define INTMAX_MIN  LLONG_MIN\n"
    "#define INTPTR_MAX  LONG_MAX\n"
    "#define INTMAX_MAX  LLONG_MAX\n"
    "#define UINTPTR_MAX ULONG_MAX\n"
    "#define UINTMAX_MAX ULLONG_MAX\n"
    "#define PTRDIFF_MIN INTPTR_MIN\n"
    "#define PTRDIFF_MAX INTPTR_MAX\n"
    "#define SIZE_MAX    UINT64_MAX\n"
    "} // namespace __jitify_stdint_ns\n"
    "namespace std { using namespace __jitify_stdint_ns; }\n"
    "using namespace __jitify_stdint_ns;\n"
    "#endif";

// TODO: offsetof
static const char* jitsafe_header_stddef_h =
    "#ifndef __STDDEF_HEADER\n"
    "#define __STDDEF_HEADER\n"
    "#include <climits>\n"
    "namespace __jitify_stddef_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "typedef decltype(nullptr) nullptr_t;\n"
    "#if defined(_MSC_VER)\n"
    "  typedef double max_align_t;\n"
    "#elif defined(__APPLE__)\n"
    "  typedef long double max_align_t;\n"
    "#else\n"
    "  // Define max_align_t to match the GCC definition.\n"
    "  typedef struct {\n"
    "    long long __jitify_max_align_nonce1\n"
    "        __attribute__((__aligned__(__alignof__(long long))));\n"
    "    long double __jitify_max_align_nonce2\n"
    "        __attribute__((__aligned__(__alignof__(long double))));\n"
    "  } max_align_t;\n"
    "#endif\n"
    "#endif  // __cplusplus >= 201103L\n"
    "#if __cplusplus >= 201703L\n"
    "enum class byte : unsigned char {};\n"
    "#endif  // __cplusplus >= 201703L\n"
    "} // namespace __jitify_stddef_ns\n"
    "namespace std {\n"
    "  // NVRTC provides built-in definitions of ::size_t and ::ptrdiff_t.\n"
    "  using ::size_t;\n"
    "  using ::ptrdiff_t;\n"
    "  using namespace __jitify_stddef_ns;\n"
    "} // namespace std\n"
    "using namespace __jitify_stddef_ns;\n"
    "#endif\n";

static const char* jitsafe_header_utility =
    "#pragma once\n"
    "namespace std {\n"
    "template<class T1, class T2>\n"
    "struct pair {\n"
    "	T1 first;\n"
    "	T2 second;\n"
    "	inline pair() {}\n"
    "	inline pair(T1 const& first_, T2 const& second_)\n"
    "		: first(first_), second(second_) {}\n"
    "	// TODO: Standard includes many more constructors...\n"
    "	// TODO: Comparison operators\n"
    "};\n"
    "template<class T1, class T2>\n"
    "pair<T1,T2> make_pair(T1 const& first, T2 const& second) {\n"
    "	return pair<T1,T2>(first, second);\n"
    "}\n"
    "}  // namespace std\n";

// TODO: incomplete
static const char* jitsafe_header_complex =
    "#pragma once\n"
    "namespace std {\n"
    "template<typename T>\n"
    "class complex {\n"
    "	T _real;\n"
    "	T _imag;\n"
    "public:\n"
    "	complex() : _real(0), _imag(0) {}\n"
    "	complex(T const& real, T const& imag)\n"
    "		: _real(real), _imag(imag) {}\n"
    "	complex(T const& real)\n"
    "               : _real(real), _imag(static_cast<T>(0)) {}\n"
    "	T const& real() const { return _real; }\n"
    "	T&       real()       { return _real; }\n"
    "	void real(const T &r) { _real = r; }\n"
    "	T const& imag() const { return _imag; }\n"
    "	T&       imag()       { return _imag; }\n"
    "	void imag(const T &i) { _imag = i; }\n"
    "       complex<T>& operator+=(const complex<T> z)\n"
    "         { _real += z.real(); _imag += z.imag(); return *this; }\n"
    "};\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs)\n"
    "  { return complex<T>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),\n"
    "                      lhs.real()*rhs.imag()+lhs.imag()*rhs.real()); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const T & rhs)\n"
    "  { return complexs<T>(lhs.real()*rhs,lhs.imag()*rhs); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const T& lhs, const complex<T>& rhs)\n"
    "  { return complexs<T>(rhs.real()*lhs,rhs.imag()*lhs); }\n"
    "}  // namespace std\n";

// TODO: This is incomplete (missing binary and integer funcs, macros,
// constants, types)
static const char* jitsafe_header_math_h =
    "#ifndef __MATH_HEADER\n"
    "#define __MATH_HEADER\n"
    "namespace __jitify_math_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/ \\\n"
    "	inline float       f(float x)          { return ::f(x); } \\\n"
    "	/*inline long double f(long double x)    { return ::f(x); }*/\n"
    "#else\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/\n"
    "#endif\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)\n"
    "template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)\n"
    "template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, "
    "exp); }\n"
    "template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, "
    "exp); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)\n"
    "template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, "
    "intpart); }\n"
    "template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)\n"
    "template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)\n"
    "template<typename T> inline T abs(T x) { return ::abs(x); }\n"
    "#if __cplusplus >= 201103L\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)\n"
    "template<typename T> inline int ilogb(T x) { return ::ilogb(x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)\n"
    "template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, "
    "n); }\n"
    "template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, "
    "n); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)\n"
    "template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(round)\n"
    "template<typename T> inline long lround(T x) { return ::lround(x); }\n"
    "template<typename T> inline long long llround(T x) { return ::llround(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)\n"
    "template<typename T> inline long lrint(T x) { return ::lrint(x); }\n"
    "template<typename T> inline long long llrint(T x) { return ::llrint(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)\n"
    // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
    // fmax, fmin, fma
    "#endif\n"
    "#undef DEFINE_MATH_UNARY_FUNC_WRAPPER\n"
    "} // namespace __jitify_math_ns\n"
    "namespace std { using namespace __jitify_math_ns; }\n"
    "#define M_PI 3.14159265358979323846\n"
    // Note: Global namespace already includes CUDA math funcs
    "//using namespace __jitify_math_ns;\n"
    "#endif\n";

static const char* jitsafe_header_algorithm = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace std {
    #if __cplusplus == 201103L
    #define JITIFY_CXX14_CONSTEXPR
    #else
    #define JITIFY_CXX14_CONSTEXPR constexpr
    #endif
    template<class T> JITIFY_CXX14_CONSTEXPR const T& max(const T& a, const T& b)
    {
      return (b > a) ? b : a;
    }
    template<class T> JITIFY_CXX14_CONSTEXPR const T& min(const T& a, const T& b)
    {
      return (b < a) ? b : a;
    }
    }  // namespace std
    #endif
 )";

static const char* jitsafe_header_assert_h = R"(
#ifndef __ASSERT_HEADER
#define __ASSERT_HEADER
#include "stddef.h"
#endif
 )";

static const char* jitsafe_header_stdio_h = R"(
#ifndef __STDIO_HEADER
#define __STDIO_HEADER
#include "stddef.h"
#endif
 )";

static const char* jitsafe_header_stdlib_h = R"(
#ifndef __STDLIB_HEADER
#define __STDLIB_HEADER
#include "stddef.h"
#endif
 )";

const std::unordered_map<std::string, std::string>& jitsafe_headers() {
    static const std::unordered_map<std::string, std::string> result = {
        {"float.h", jitsafe_header_float_h},
        {"limits.h", jitsafe_header_limits_h},
        {"stdint.h", jitsafe_header_stdint_h},
        {"stddef.h", jitsafe_header_stddef_h},
        {"stdio.h", jitsafe_header_stdio_h},
        {"iterator", jitsafe_header_iterator},
        {"limits", jitsafe_header_limits},
        {"type_traits", jitsafe_header_type_traits},
        {"utility", jitsafe_header_utility},
        {"math.h", jitsafe_header_math_h},
        {"complex", jitsafe_header_complex},
        {"algorithm", jitsafe_header_algorithm},
        {"stdlib.h", jitsafe_header_stdlib_h},
        {"assert.h", jitsafe_header_assert_h},

        {"cfloat", jitsafe_header_float_h},
        {"cassert", jitsafe_header_assert_h},
        {"cstdlib", jitsafe_header_stdlib_h},
        {"cmath", jitsafe_header_math_h},
        {"cstdio", jitsafe_header_stdio_h},
        {"cstddef", jitsafe_header_stddef_h},
        {"cstdint", jitsafe_header_stdint_h},
        {"climits", jitsafe_header_limits_h},
    };

    return result;
}

}  // namespace kernel_launcher