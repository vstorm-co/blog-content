# Model Comparison Results

## Structured Extraction Comparison

| Field | GPT-OSS 20B | GPT-OSS 120B | Kimi K2 Instruct | Llama 4 Maverick | Llama 4 Scout |
|-------|-------------|--------------|------------------|------------------|---------------|
| **Success** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Name** | IBM Personal Computer Model 5150 | IBM Personal Computer 5150 | IBM Personal Computer 5150 | IBM Personal Computer | IBM Personal Computer model 5150 |
| **Year Start** | 1981 | 1981 | 1981 | 1981 | 1981 |
| **Year End** | 1987 | 1987 | 1987 | 1987 | 1987 |
| **Category** | Personal Computer | Computer | Computing Hardware | Computing | Computer |
| **Summary** | IBM PC 5150, launched 1981, Intel 8088, 16KB RAM, open architecture, 5 slots, CGA, 5.25" floppy, set standard for PC industry. | The IBM Personal Computer model 5150, launched in 1981, featured an Intel 8088 CPU, 16 KB RAM, five expansion slots and CGA graphics, setting the PC standard and spawning a clone market. | The IBM Personal Computer model 5150, launched in August 1981, transformed the computing landscape with its open-architecture Intel 8088-based design running at 4.77 MHz. Starting with 16 KB of expandable RAM, five expansion slots, CGA graphics and optional 5.25-inch floppy drives, the beige modular system encouraged third-party hardware and software, exceeded projected 250,000-unit sales within a year, and established enduring PC standards while competing against the entrenched Apple II. | The IBM Personal Computer, model 5150, was a groundbreaking machine that revolutionized the computing industry with its open architecture, Intel 8088 processor, and modular design, making it accessible to businesses and enthusiasts alike, and establishing standards that persist in modern computing. | The IBM Personal Computer, model 5150, launched in 1981, revolutionized the computing industry with its Intel 8088 processor, open architecture, and modular design. |
| **Summary Length** | 129 chars | 163 chars | 427 chars ⚠️ | 238 chars ⚠️ | 145 chars |
| **Keywords** | IBM PC, Intel 8088, CGA graphics, open architecture, clones, floppy drive, 1981 | IBM PC, model 5150, Intel 8088, 1981, personal computer, PC compatible | IBM 5150, personal computer, open architecture, Intel 8088, PC expansion bus | IBM, Personal Computer, 5150, Intel 8088, open architecture | IBM Personal Computer, model 5150, Intel 8088 processor, CGA graphics, floppy disk drives, expansion slots |
| **Keywords Count** | 7 | 6 | 5 | 5 | 6 |
| **ID Number** | IBM-PC-5150-SYS-001-1981-US | IBM-PC-5150-SYS-001-1981-US | IBM-PC-5150-SYS-001-1981-US | IBM-PC-5150-SYS-001-1981-US | IBM-PC-5150-SYS-001-1981-US |
| **Tables Count** | 1 | 1 | 1 | 1 | 1 |
| **Table Title** | Event Timeline | IBM 5150 Timeline | Key Events | Event Timeline | Key Dates |
| **Table Format** | Object | Object | Object | Object | Array ⚠️ |

## Notes

- ⚠️ **Summary Length Violations**: Kimi K2 (427 chars) and Llama 4 Maverick (238 chars) exceed the 200 character maximum
- ⚠️ **Table Format**: Llama 4 Scout used array format instead of object format for table data
- All models successfully extracted the identification number
- All models correctly identified the year range (1981-1987)
