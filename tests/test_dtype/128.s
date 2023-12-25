
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_10n53>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[6:7], s[4:5], 1                               // 000000001614: 84868104
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	s_add_u32 s0, s0, s15                                      // 000000001624: 80000F00
	global_load_u16 v1, v0, s[2:3]                             // 000000001628: DC4A0000 01020000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt vmcnt(0)                                         // 000000001634: BF8903F7
	global_store_b8 v0, v1, s[0:1]                             // 000000001638: DC620000 00000100
	s_nop 0                                                    // 000000001640: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001644: BFB60003
	s_endpgm                                                   // 000000001648: BFB00000
