
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4n66>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	v_mov_b32_e32 v2, 0                                        // 000000001608: 7E040280
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001610: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001618: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s6, s6, s15                                      // 000000001620: 80060F06
	s_addc_u32 s7, s7, s3                                      // 000000001624: 82070307
	s_lshl_b64 s[2:3], s[2:3], 3                               // 000000001628: 84828302
	global_load_i8 v0, v2, s[6:7]                              // 00000000162C: DC460000 00060002
	s_waitcnt vmcnt(0)                                         // 000000001634: BF8903F7
	v_readfirstlane_b32 s6, v0                                 // 000000001638: 7E0C0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000163C: BF870491
	s_sext_i32_i16 s6, s6                                      // 000000001640: BE860F06
	s_ashr_i32 s7, s6, 31                                      // 000000001644: 86079F06
	s_add_u32 s0, s0, s2                                       // 000000001648: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000164C: 82010301
	s_load_b64 s[0:1], s[0:1], null                            // 000000001650: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001658: BF89FC07
	s_mul_hi_u32 s8, s0, s6                                    // 00000000165C: 96880600
	s_mul_i32 s7, s0, s7                                       // 000000001660: 96070700
	s_mul_i32 s1, s1, s6                                       // 000000001664: 96010601
	s_add_i32 s7, s8, s7                                       // 000000001668: 81070708
	s_mul_i32 s0, s0, s6                                       // 00000000166C: 96000600
	s_add_i32 s7, s7, s1                                       // 000000001670: 81070107
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001674: BF870009
	v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s7             // 000000001678: CA100000 00000007
	s_add_u32 s0, s4, s2                                       // 000000001680: 80000204
	s_addc_u32 s1, s5, s3                                      // 000000001684: 82010305
	global_store_b64 v2, v[0:1], s[0:1]                        // 000000001688: DC6E0000 00000002
	s_nop 0                                                    // 000000001690: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001694: BFB60003
	s_endpgm                                                   // 000000001698: BFB00000
