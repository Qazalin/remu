
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2_2n12>:
	s_lshl_b32 s2, s14, 1                                      // 000000001600: 8402810E
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_add_i32 s2, s2, s15                                      // 00000000160C: 81020F02
	s_mov_b32 s5, 0                                            // 000000001610: BE850080
	s_mul_hi_i32 s3, s2, 0x55555556                            // 000000001614: 9703FF02 55555556
	v_dual_mov_b32 v1, s5 :: v_dual_mov_b32 v2, 0              // 00000000161C: CA100005 01020080
	s_lshr_b32 s4, s3, 31                                      // 000000001624: 85049F03
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001628: BF870499
	s_add_i32 s3, s3, s4                                       // 00000000162C: 81030403
	s_mul_i32 s3, s3, 3                                        // 000000001630: 96038303
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001634: BF870499
	s_sub_i32 s2, s2, s3                                       // 000000001638: 81820302
	s_cmp_lt_i32 s2, 1                                         // 00000000163C: BF048102
	s_cselect_b32 s4, -1, 0                                    // 000000001640: 980480C1
	s_lshl_b32 s2, s15, 1                                      // 000000001644: 8402810F
	v_cndmask_b32_e64 v0, 0, 1, s4                             // 000000001648: D5010000 00110280
	s_ashr_i32 s3, s2, 31                                      // 000000001650: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001654: BF8704D9
	s_lshl_b64 s[2:3], s[2:3], 3                               // 000000001658: 84828302
	s_waitcnt lgkmcnt(0)                                       // 00000000165C: BF89FC07
	s_add_u32 s2, s0, s2                                       // 000000001660: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001664: 82030301
	s_ashr_i32 s15, s14, 31                                    // 000000001668: 860F9F0E
	s_lshl_b64 s[0:1], s[14:15], 3                             // 00000000166C: 8480830E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001670: BF870009
	s_add_u32 s0, s2, s0                                       // 000000001674: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001678: 82010103
	global_store_b64 v2, v[0:1], s[0:1]                        // 00000000167C: DC6E0000 00000002
	s_nop 0                                                    // 000000001684: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001688: BFB60003
	s_endpgm                                                   // 00000000168C: BFB00000
