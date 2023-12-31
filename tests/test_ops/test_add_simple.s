
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_256>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_endpgm                                                   // 00000000166C: BFB00000
