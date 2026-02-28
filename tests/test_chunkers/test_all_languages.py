"""Comprehensive tests for all 8 supported languages in AST chunking."""

import pytest

from codii.chunkers.ast_chunker import ASTChunker, CodeChunk


class TestChunkPython:
    """Comprehensive tests for Python code chunking."""

    def test_function_extraction(self):
        """Extract Python functions with correct names."""
        chunker = ASTChunker()

        content = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True

def another_function(x, y):
    return x + y
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 2

        names = [c.name for c in function_chunks if c.name]
        assert "hello_world" in names
        assert "another_function" in names

    def test_class_extraction(self):
        """Extract Python classes with correct names."""
        chunker = ASTChunker()

        content = '''
class MyClass:
    def __init__(self):
        self.value = 10

    def get_value(self):
        return self.value

class AnotherClass:
    pass
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 2

        names = [c.name for c in class_chunks if c.name]
        assert "MyClass" in names
        assert "AnotherClass" in names

    def test_async_function(self):
        """Extract async functions."""
        chunker = ASTChunker()

        content = '''
async def fetch_data():
    """Fetch data asynchronously."""
    await some_operation()
    return data

async def process_items(items):
    for item in items:
        await process(item)
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Should find async functions
        async_chunks = [c for c in chunks if "function" in c.chunk_type]
        assert len(async_chunks) >= 2

    def test_nested_structures(self):
        """Handle nested functions and classes."""
        chunker = ASTChunker()

        content = '''
class OuterClass:
    def outer_method(self):
        def inner_function():
            pass
        return inner_function

    class InnerClass:
        pass
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        # Should extract the outer class at minimum
        assert len(chunks) >= 1

    def test_line_numbers(self):
        """Verify line number accuracy."""
        chunker = ASTChunker()

        content = '''# Line 1

def my_function():
    """A function starting at line 3."""
    pass

class MyClass:
    """A class starting at line 7."""
    pass
'''
        chunks = chunker.chunk_file(content, "/test/file.py", "python")

        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_syntax_error_fallback(self):
        """Handle files with syntax errors."""
        chunker = ASTChunker()

        # Invalid Python syntax
        content = "def broken(:\n    pass"

        # Should not raise, may fallback to text chunking
        chunks = chunker.chunk_file(content, "/test/broken.py", "python")
        assert isinstance(chunks, list)


class TestChunkJavaScript:
    """Comprehensive tests for JavaScript code chunking."""

    def test_function_extraction(self):
        """Extract JavaScript function declarations."""
        chunker = ASTChunker()

        content = '''
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

function anotherFunction(x, y) {
    return x + y;
}
'''
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        assert len(chunks) >= 2

    def test_class_extraction(self):
        """Extract JavaScript classes."""
        chunker = ASTChunker()

        content = '''
class MyClass {
    constructor() {
        this.value = 10;
    }

    getValue() {
        return this.value;
    }
}

class AnotherClass extends MyClass {
    method() {
        return super.getValue();
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1

    def test_arrow_function(self):
        """Extract arrow functions."""
        chunker = ASTChunker()

        content = '''
const arrowFunc = () => {
    return 42;
};

const add = (a, b) => a + b;

const obj = {
    method: () => {
        return "method";
    }
};
'''
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        # Arrow functions should be extracted
        assert len(chunks) >= 1

    def test_method_extraction(self):
        """Extract class methods."""
        chunker = ASTChunker()

        content = '''
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }

    multiply(a, b) {
        return a * b;
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/file.js", "javascript")

        # Should extract the class
        assert len(chunks) >= 1


class TestChunkTypeScript:
    """Comprehensive tests for TypeScript code chunking."""

    def test_function_extraction(self):
        """Extract TypeScript functions with type annotations."""
        chunker = ASTChunker()

        content = '''
function greet(name: string): string {
    return `Hello, ${name}!`;
}

const add = (a: number, b: number): number => a + b;
'''
        chunks = chunker.chunk_file(content, "/test/file.ts", "typescript")

        assert len(chunks) >= 1

    def test_interface_extraction(self):
        """Extract TypeScript interfaces."""
        chunker = ASTChunker()

        content = '''
interface User {
    name: string;
    age: number;
    email?: string;
}

interface Product {
    id: number;
    name: string;
    price: number;
}
'''
        chunks = chunker.chunk_file(content, "/test/file.ts", "typescript")

        # Should extract interfaces
        assert len(chunks) >= 1

    def test_type_alias(self):
        """Extract TypeScript type aliases."""
        chunker = ASTChunker()

        content = '''
type UserID = string | number;

type Point = {
    x: number;
    y: number;
};

type Callback = (data: unknown) => void;
'''
        chunks = chunker.chunk_file(content, "/test/file.ts", "typescript")

        # Should extract type aliases
        assert len(chunks) >= 1

    def test_class_with_types(self):
        """Extract TypeScript classes with type annotations."""
        chunker = ASTChunker()

        content = '''
class UserService<T extends User> {
    private users: T[] = [];

    add(user: T): void {
        this.users.push(user);
    }

    get(id: UserID): T | undefined {
        return this.users.find(u => u.id === id);
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/file.ts", "typescript")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1


class TestChunkGo:
    """Comprehensive tests for Go code chunking."""

    def test_function_extraction(self):
        """Extract Go functions."""
        chunker = ASTChunker()

        content = '''
package main

import "fmt"

func helloWorld() {
    fmt.Println("Hello, World!")
}

func add(a, b int) int {
    return a + b
}

func main() {
    helloWorld()
}
'''
        chunks = chunker.chunk_file(content, "/test/file.go", "go")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 3

    def test_method_extraction(self):
        """Extract Go methods."""
        chunker = ASTChunker()

        content = '''
package main

type Calculator struct {
    result int
}

func (c *Calculator) Add(a int) {
    c.result += a
}

func (c *Calculator) GetResult() int {
    return c.result
}
'''
        chunks = chunker.chunk_file(content, "/test/file.go", "go")

        # Should extract methods
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) >= 2

    def test_struct_extraction(self):
        """Extract Go structs (type declarations)."""
        chunker = ASTChunker()

        content = '''
package main

type User struct {
    Name string
    Age  int
}

type Point struct {
    X, Y float64
}
'''
        chunks = chunker.chunk_file(content, "/test/file.go", "go")

        # Should extract type declarations (structs)
        type_chunks = [c for c in chunks if c.chunk_type == "type"]
        assert len(type_chunks) >= 2

    def test_line_numbers(self):
        """Verify line number accuracy for Go."""
        chunker = ASTChunker()

        content = '''package main

func main() {
    println("hello")
}
'''
        chunks = chunker.chunk_file(content, "/test/file.go", "go")

        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line


class TestChunkRust:
    """Comprehensive tests for Rust code chunking."""

    def test_function_extraction(self):
        """Extract Rust functions."""
        chunker = ASTChunker()

        content = '''
fn hello_world() {
    println!("Hello, World!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn public_function() -> bool {
    true
}
'''
        chunks = chunker.chunk_file(content, "/test/file.rs", "rust")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 3

    def test_struct_extraction(self):
        """Extract Rust structs."""
        chunker = ASTChunker()

        content = '''
struct Point {
    x: f64,
    y: f64,
}

struct User {
    name: String,
    age: u32,
}
'''
        chunks = chunker.chunk_file(content, "/test/file.rs", "rust")

        struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
        assert len(struct_chunks) >= 2

    def test_enum_extraction(self):
        """Extract Rust enums."""
        chunker = ASTChunker()

        content = '''
enum Color {
    Red,
    Green,
    Blue,
}

enum Option<T> {
    Some(T),
    None,
}
'''
        chunks = chunker.chunk_file(content, "/test/file.rs", "rust")

        enum_chunks = [c for c in chunks if c.chunk_type == "enum"]
        assert len(enum_chunks) >= 1

    def test_impl_extraction(self):
        """Extract Rust impl blocks."""
        chunker = ASTChunker()

        content = '''
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/file.rs", "rust")

        impl_chunks = [c for c in chunks if c.chunk_type == "impl"]
        assert len(impl_chunks) >= 1

    def test_trait_extraction(self):
        """Extract Rust traits."""
        chunker = ASTChunker()

        content = '''
trait Drawable {
    fn draw(&self);
}

trait Shape {
    fn area(&self) -> f64;
}
'''
        chunks = chunker.chunk_file(content, "/test/file.rs", "rust")

        trait_chunks = [c for c in chunks if c.chunk_type == "trait"]
        assert len(trait_chunks) >= 2


class TestChunkJava:
    """Comprehensive tests for Java code chunking."""

    def test_method_extraction(self):
        """Extract Java methods."""
        chunker = ASTChunker()

        content = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    private int subtract(int a, int b) {
        return a - b;
    }

    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/Calculator.java", "java")

        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) >= 3

    def test_class_extraction(self):
        """Extract Java classes."""
        chunker = ASTChunker()

        content = '''
public class User {
    private String name;
    private int age;

    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }
}

class Product {
    private int id;
    private String name;
}
'''
        chunks = chunker.chunk_file(content, "/test/User.java", "java")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 2

    def test_interface_extraction(self):
        """Extract Java interfaces."""
        chunker = ASTChunker()

        content = '''
public interface Drawable {
    void draw();
}

interface Shape {
    double area();
}
'''
        chunks = chunker.chunk_file(content, "/test/Drawable.java", "java")

        interface_chunks = [c for c in chunks if c.chunk_type == "interface"]
        assert len(interface_chunks) >= 2

    def test_enum_extraction(self):
        """Extract Java enums."""
        chunker = ASTChunker()

        content = '''
public enum Color {
    RED,
    GREEN,
    BLUE;

    public String toHex() {
        switch (this) {
            case RED: return "#FF0000";
            case GREEN: return "#00FF00";
            case BLUE: return "#0000FF";
        }
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/Color.java", "java")

        enum_chunks = [c for c in chunks if c.chunk_type == "enum"]
        assert len(enum_chunks) >= 1


class TestChunkC:
    """Comprehensive tests for C code chunking."""

    def test_function_extraction(self):
        """Extract C functions."""
        chunker = ASTChunker()

        content = '''
#include <stdio.h>

void hello_world() {
    printf("Hello, World!\\n");
}

int add(int a, int b) {
    return a + b;
}

int main() {
    hello_world();
    return 0;
}
'''
        chunks = chunker.chunk_file(content, "/test/file.c", "c")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 3

    def test_struct_extraction(self):
        """Extract C structs."""
        chunker = ASTChunker()

        content = '''
struct Point {
    double x;
    double y;
};

struct User {
    char name[100];
    int age;
};
'''
        chunks = chunker.chunk_file(content, "/test/file.c", "c")

        struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
        assert len(struct_chunks) >= 2

    def test_enum_extraction(self):
        """Extract C enums."""
        chunker = ASTChunker()

        content = '''
enum Color {
    RED,
    GREEN,
    BLUE
};

enum Status {
    OK = 0,
    ERROR = 1,
    PENDING = 2
};
'''
        chunks = chunker.chunk_file(content, "/test/file.c", "c")

        enum_chunks = [c for c in chunks if c.chunk_type == "enum"]
        assert len(enum_chunks) >= 2

    def test_typedef_struct(self):
        """Handle typedef structs."""
        chunker = ASTChunker()

        content = '''
typedef struct {
    int x;
    int y;
} Point;

typedef struct Node {
    int value;
    struct Node* next;
} Node;
'''
        chunks = chunker.chunk_file(content, "/test/file.c", "c")

        # Should extract structs
        assert len(chunks) >= 1


class TestChunkCpp:
    """Comprehensive tests for C++ code chunking."""

    def test_function_extraction(self):
        """Extract C++ functions."""
        chunker = ASTChunker()

        content = '''
#include <iostream>

void hello_world() {
    std::cout << "Hello, World!" << std::endl;
}

int add(int a, int b) {
    return a + b;
}

template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
'''
        chunks = chunker.chunk_file(content, "/test/file.cpp", "cpp")

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) >= 3

    def test_class_extraction(self):
        """Extract C++ classes."""
        chunker = ASTChunker()

        content = '''
class Point {
private:
    double x, y;

public:
    Point(double x, double y) : x(x), y(y) {}

    double distance() const {
        return sqrt(x*x + y*y);
    }
};

class User {
    std::string name;
    int age;
public:
    User(const std::string& n, int a) : name(n), age(a) {}
};
'''
        chunks = chunker.chunk_file(content, "/test/file.cpp", "cpp")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 2

    def test_struct_extraction(self):
        """Extract C++ structs."""
        chunker = ASTChunker()

        content = '''
struct Point {
    double x;
    double y;
};

struct Color {
    uint8_t r, g, b;
};
'''
        chunks = chunker.chunk_file(content, "/test/file.cpp", "cpp")

        struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
        assert len(struct_chunks) >= 2

    def test_namespace_extraction(self):
        """Extract C++ namespaces."""
        chunker = ASTChunker()

        content = '''
namespace math {
    double pi = 3.14159;

    double square(double x) {
        return x * x;
    }
}

namespace utils {
    void log(const std::string& msg) {
        std::cout << msg << std::endl;
    }
}
'''
        chunks = chunker.chunk_file(content, "/test/file.cpp", "cpp")

        namespace_chunks = [c for c in chunks if c.chunk_type == "namespace"]
        assert len(namespace_chunks) >= 2


class TestAllLanguagesSupport:
    """Verify all 8 languages are supported."""

    def test_all_languages_supported(self):
        """Check that all 8 languages report as supported."""
        chunker = ASTChunker()

        supported_languages = [
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "c",
            "cpp",
        ]

        for lang in supported_languages:
            assert chunker.is_language_supported(lang), f"{lang} should be supported"

    def test_unsupported_language_fallback(self):
        """Unsupported languages should fallback to text chunking."""
        chunker = ASTChunker()

        content = "Some text content for an unsupported language."
        chunks = chunker.chunk_file(content, "/test/file.xyz", "ruby")

        # Should fallback to text chunking and return something
        assert isinstance(chunks, list)